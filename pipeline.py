import cv2
import numpy as np
import torch
from torchvision import transforms
import tensorflow as tf
import tensorflow_hub as hub
from u2net import U2NET  # Import the U^2-Net model class
import math


# Function to load the U^2-Net model
def load_u2net_model(model_path):
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    net.eval()
    return net


# Function to perform background removal using U^2-Net
def remove_background(image, model):
    # Transform the image
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
    mask = prediction > 0.501  # Lowered threshold

    # Resize mask to original image size
    mask = cv2.resize(
        mask.astype(np.uint8),
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Apply mask to the image
    result = cv2.bitwise_and(image, image, mask=mask)
    return result, mask


# Function to load MoveNet MultiPose model
def load_movenet_model():
    model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
    return model


# Function to run pose estimation
def run_pose_estimation(image, model):
    input_size = 256
    image_resized = tf.image.resize_with_pad(image, input_size, input_size)
    input_image = tf.cast(image_resized, dtype=tf.int32)
    input_image = tf.expand_dims(input_image, axis=0)
    outputs = model.signatures["serving_default"](input_image)
    keypoints_with_scores = outputs["output_0"].numpy()
    return keypoints_with_scores


# Function to rotate an image
def rotate_image(image, angle):
    # Get the image size
    h, w = image.shape[:2]
    # Calculate the center of the image
    center = (w / 2, h / 2)
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Perform the rotation
    rotated_image = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated_image


# Main function
def main():
    # Paths to images and models
    input_image_path = "test.jpg"
    background_image_path = "background.jpg"
    hat_image_path = "hat.png"
    u2net_model_path = "u2net/u2net.pth"

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

    # Resize background to match original image
    background_image = cv2.resize(
        background_image, (original_image.shape[1], original_image.shape[0])
    )

    # Remove background
    print("Removing background...")
    u2net_model = load_u2net_model(u2net_model_path)
    person_segmented, mask = remove_background(original_image, u2net_model)

    # Replace background
    mask_3ch = cv2.merge([mask, mask, mask])
    person_with_new_bg = np.where(mask_3ch == 1, person_segmented, background_image)

    # Convert image to RGB for TensorFlow
    image_rgb = cv2.cvtColor(person_with_new_bg, cv2.COLOR_BGR2RGB)
    image_tf = tf.convert_to_tensor(image_rgb, dtype=tf.uint8)

    # Load MoveNet model
    print("Loading MoveNet model...")
    movenet_model = load_movenet_model()

    # Run pose estimation
    print("Running pose estimation...")
    keypoints_with_scores = run_pose_estimation(image_tf, movenet_model)

    # Add hats to each detected person
    height, width, _ = person_with_new_bg.shape
    for person in keypoints_with_scores[0]:
        keypoints = person[:51].reshape((17, 3))
        scores = keypoints[:, 2]
        keypoints_xy = keypoints[:, :2]
        if scores[0] < 0.3:
            continue  # Skip person if confidence is low

        # Coordinates are in [y, x] format
        # Get keypoints positions
        nose = keypoints_xy[0]
        left_eye = keypoints_xy[1]
        right_eye = keypoints_xy[2]
        left_eye_score = scores[1]
        right_eye_score = scores[2]

        if left_eye_score > 0.3 and right_eye_score > 0.3:
            # Use the midpoint between left and right eyes for head center
            head_x = (left_eye[1] + right_eye[1]) / 2 * width
            head_y = (left_eye[0] + right_eye[0]) / 2 * height

            # Calculate the angle of the head
            delta_x = right_eye[1] * width - left_eye[1] * width
            delta_y = right_eye[0] * height - left_eye[0] * height
            angle_deg = -math.degrees(math.atan2(delta_y, delta_x))

        else:
            # Use nose position if eyes are not detected
            head_x = nose[1] * width
            head_y = nose[0] * height
            angle_deg = 0  # No rotation

        # Adjust y-coordinate to place the hat on top
        head_y -= int(0.15 * height)  # Adjust this value as needed

        # Draw circle at head center for debugging
        # cv2.circle(person_with_new_bg, (int(head_x), int(head_y)), 5, (0, 0, 255), -1)

        # Resize hat
        hat_width = int(width * 0.2)
        hat_height = int(hat_width * hat_image.shape[0] / hat_image.shape[1])
        resized_hat = cv2.resize(
            hat_image, (hat_width, hat_height), interpolation=cv2.INTER_AREA
        )

        # Rotate hat
        rotated_hat = rotate_image(resized_hat, angle_deg - 180)

        # Get dimensions of the rotated hat
        hat_h, hat_w = rotated_hat.shape[0], rotated_hat.shape[1]

        # Calculate the roll angle (tilt)
        roll_offset = int(math.sin(math.radians(angle_deg)) * hat_h / 2)

        # Adjust hat position based on roll angle
        x1 = int(head_x - hat_w / 2 + roll_offset)
        y1 = int(head_y - hat_h)

        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x1 + hat_w)
        y2 = min(height, y1 + hat_h)

        # Adjust hat size if it goes out of bounds
        rotated_hat = rotated_hat[0 : (y2 - y1), 0 : (x2 - x1)]

        # Extract hat alpha channel and overlay hat
        if rotated_hat.shape[2] == 4:
            alpha_hat = rotated_hat[:, :, 3] / 255.0
            alpha_bg = 1.0 - alpha_hat
            for c in range(0, 3):
                person_with_new_bg[y1:y2, x1:x2, c] = (
                    alpha_hat * rotated_hat[:, :, c]
                    + alpha_bg * person_with_new_bg[y1:y2, x1:x2, c]
                )
        else:
            person_with_new_bg[y1:y2, x1:x2] = rotated_hat[:, :, :3]

    # Display the result
    cv2.imshow("Result", person_with_new_bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
