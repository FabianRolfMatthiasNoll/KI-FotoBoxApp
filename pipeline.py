import cv2
import numpy as np
import mediapipe as mp
import math
import torch
from torchvision import transforms
from u2net import U2NET


# Function to load the U^2-Net model
def load_u2net_model(model_path):
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    net.eval()
    return net


# Function to perform background removal using U^2-Net
def remove_background(image, model):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # Normalization per ImageNet standards
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        d1, _, _, _, _, _, _ = model(input_tensor)
        prediction = d1[:, 0, :, :]
        prediction = torch.sigmoid(prediction)
        prediction = prediction.squeeze().cpu().numpy()

    # Adjust threshold
    mask = prediction > 0.501

    # Resize mask to original image size
    mask = cv2.resize(
        mask.astype(np.uint8),
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    return mask


# Function to detect people using YOLOv5
def detect_people_yolo(image, model, conf_threshold=0.5, debug_mode=False):
    results = model(image)

    # Extract bounding boxes from results
    detections = (
        results.xyxy[0].cpu().numpy()
    )  # Bounding boxes in [x1, y1, x2, y2, conf, class]
    bounding_boxes = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf > conf_threshold and int(cls) == 0:  # Class 0 corresponds to 'person'
            bounding_boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
            if debug_mode:
                print(
                    f"Detected person: x={x1}, y={y1}, w={x2 - x1}, h={y2 - y1}, conf={conf}"
                )

    return bounding_boxes


# Function to rotate an image
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated_image


# Function to process a single person with MediaPipe Pose
def process_person(image, bounding_box, hat_image, debug_mode=False):
    x, y, w, h = bounding_box
    person_crop = image[y : y + h, x : x + w]

    # Run MediaPipe Pose on the cropped person
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)
    results = pose.process(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
    pose.close()

    if not results.pose_landmarks:
        if debug_mode:
            print(f"No landmarks detected for bounding box: x={x}, y={y}, w={w}, h={h}")
        return image  # Skip if no landmarks are detected

    # Extract keypoints
    landmarks = results.pose_landmarks.landmark
    keypoints = np.array([[lmk.x * w, lmk.y * h, lmk.visibility] for lmk in landmarks])

    # Extract keypoint indices for the hat placement
    NOSE = mp.solutions.pose.PoseLandmark.NOSE.value
    LEFT_EYE = mp.solutions.pose.PoseLandmark.LEFT_EYE.value
    RIGHT_EYE = mp.solutions.pose.PoseLandmark.RIGHT_EYE.value

    nose = keypoints[NOSE]
    left_eye = keypoints[LEFT_EYE]
    right_eye = keypoints[RIGHT_EYE]

    # Confidence thresholds
    min_confidence = 0.5
    if left_eye[2] > min_confidence and right_eye[2] > min_confidence:
        head_x = (left_eye[0] + right_eye[0]) / 2 + x
        head_y = (left_eye[1] + right_eye[1]) / 2 + y

        # Calculate head tilt angle
        delta_x = right_eye[0] - left_eye[0]
        delta_y = right_eye[1] - left_eye[1]
        angle_deg = -math.degrees(math.atan2(delta_y, delta_x))

        # Estimate face size
        eye_distance = np.linalg.norm(left_eye[:2] - right_eye[:2])
        face_size = eye_distance * 1.5

        # Adjust hat placement vertically
        head_y -= int(face_size * 0.4)
    elif nose[2] > min_confidence:
        head_x, head_y = nose[0] + x, nose[1] + y
        face_size = w * 0.1
        angle_deg = 0
    else:
        if debug_mode:
            print(
                f"Skipping bounding box x={x}, y={y}, w={w}, h={h} due to low confidence."
            )
        return image  # Skip if no reliable keypoints

    # Resize and rotate the hat
    hat_width = int(face_size * 2.5)
    hat_height = int(hat_width * hat_image.shape[0] / hat_image.shape[1])
    resized_hat = cv2.resize(
        hat_image, (hat_width, hat_height), interpolation=cv2.INTER_AREA
    )
    rotated_hat = rotate_image(resized_hat, angle_deg - 180)

    # Place the hat on the head
    x1 = int(head_x - hat_width / 2)
    y1 = int(head_y - hat_height)
    x2 = x1 + hat_width
    y2 = y1 + hat_height

    # Ensure placement is within bounds
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])
    rotated_hat = rotated_hat[: y2 - y1, : x2 - x1]

    # Overlay the hat
    if rotated_hat.shape[2] == 4:
        alpha_hat = rotated_hat[:, :, 3].astype(np.float32) / 255.0
        alpha_bg = 1.0 - alpha_hat
        for c in range(0, 3):
            image[y1:y2, x1:x2, c] = (
                alpha_hat * rotated_hat[:, :, c].astype(np.float32)
                + alpha_bg * image[y1:y2, x1:x2, c].astype(np.float32)
            ).astype(np.uint8)
    else:
        image[y1:y2, x1:x2] = rotated_hat[:, :, :3]

    return image


# Main function
def main():
    input_image_path = "nine-people.jpg"
    background_image_path = "background.jpg"
    hat_image_path = "hat.png"
    u2net_model_path = "u2net/u2net.pth"
    debug_mode = True  # Enable debug mode

    # Load images
    original_image = cv2.imread(input_image_path)
    if original_image is None:
        print("Error: Could not read the input image.")
        return
    background_image = cv2.imread(background_image_path)
    if background_image is None:
        print("Error: Could not read the background image.")
        return
    hat_image = cv2.imread(hat_image_path, cv2.IMREAD_UNCHANGED)
    if hat_image is None:
        print("Error: Could not read the hat image.")
        return

    # Load YOLOv5 model
    print("Loading YOLOv5 model...")
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

    # Detect people using YOLOv5
    print("Detecting people with YOLOv5...")
    bounding_boxes = detect_people_yolo(original_image, model, debug_mode=debug_mode)

    # Use U^2-Net for background removal
    u2net_model = load_u2net_model(u2net_model_path)
    mask = remove_background(original_image, u2net_model)
    background_removed = cv2.bitwise_and(original_image, original_image, mask=mask)

    # Overlay the background
    background_resized = cv2.resize(
        background_image, (original_image.shape[1], original_image.shape[0])
    )
    final_image = np.where(mask[..., None] == 1, background_removed, background_resized)

    # Process each detected person using MediaPipe Pose
    for i, box in enumerate(bounding_boxes):
        print(
            f"Processing person {i + 1} in bounding box x={box[0]}, y={box[1]}, w={box[2]}, h={box[3]}..."
        )
        final_image = process_person(final_image, box, hat_image, debug_mode=True)

    # Display the final image
    cv2.imshow("Result", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
