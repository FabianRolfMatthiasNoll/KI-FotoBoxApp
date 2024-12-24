import math
import cv2
from dotenv import load_dotenv
import numpy as np
import mediapipe as mp
import os
from torchvision import transforms
import torch
from openai_client import OpenAI_Client
from u2net import U2NET


# Load U^2-Net model
def load_u2net_model(model_path):
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_path, map_location=torch.device("cuda")))
    net.eval()
    return net


def overlay_image(background, overlay, x, y):
    """
    Overlays an image with transparency onto a background image.

    Args:
        background (np.ndarray): The background image.
        overlay (np.ndarray): The overlay image with an alpha channel.
        x (int): The x-coordinate for the top-left corner of the overlay.
        y (int): The y-coordinate for the top-left corner of the overlay.

    Returns:
        np.ndarray: The resulting image with the overlay applied.
    """
    h, w = overlay.shape[:2]
    alpha = overlay[:, :, 3] / 255.0

    # Adjust for out-of-bounds coordinates
    y_start = max(0, y)
    y_end = min(background.shape[0], y + h)
    x_start = max(0, x)
    x_end = min(background.shape[1], x + w)

    # Calculate the visible portion of the overlay
    overlay_y_start = max(0, -y)
    overlay_y_end = overlay_y_start + (y_end - y_start)
    overlay_x_start = max(0, -x)
    overlay_x_end = overlay_x_start + (x_end - x_start)

    for c in range(3):  # Iterate over RGB channels
        background[y_start:y_end, x_start:x_end, c] = (
            alpha[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end]
            * overlay[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end, c]
            + (1 - alpha[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end])
            * background[y_start:y_end, x_start:x_end, c]
        )
    return background


def preprocess_image(image):
    """
    Simplifies image preprocessing by normalizing brightness without over-enhancing.
    """
    return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def apply_background(image, background_path):
    """
    Replaces the background of the image using U²-Net for segmentation.
    """
    # Load and apply U²-Net for background removal
    u2net_model = load_u2net_model("u2net/u2net.pth")
    mask = remove_background(image, u2net_model)

    # Load the new background
    background = cv2.imread(background_path)
    if background is None:
        raise ValueError(f"Background not found: {background_path}")
    background = cv2.resize(background, (image.shape[1], image.shape[0]))

    # Ensure mask is binary
    mask = (mask * 255).astype(np.uint8)
    mask_inv = cv2.bitwise_not(mask)

    # Extract foreground and background
    foreground = cv2.bitwise_and(image, image, mask=mask)
    new_background = cv2.bitwise_and(background, background, mask=mask_inv)

    # Combine the two images
    combined = cv2.add(foreground, new_background)
    return combined


# Background removal using U^2-Net
def remove_background(image, model):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        d1, *_ = model(input_tensor)
        prediction = torch.sigmoid(d1[:, 0, :, :]).squeeze().cpu().numpy()
    mask = (prediction > 0.8).astype(np.uint8)
    return cv2.resize(
        mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
    )


def apply_accessories(image, accessories, asset_dirs):
    """
    Applies accessories (hats, glasses, beards) to all faces detected using MediaPipe FaceMesh.
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=20,
        min_detection_confidence=0.4,
    )
    h, w, _ = image.shape
    annotated_image = image.copy()

    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        print("No faces detected.")
        return image

    print(f"Number of faces detected: {len(results.multi_face_landmarks)}")
    for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
        print(f"Processing face {face_id + 1}")

        # Hat placement
        if "hat" in accessories and accessories["hat"] != "none":
            hat_path = os.path.join(asset_dirs["hats"], accessories["hat"] + ".png")
            hat_image = cv2.imread(hat_path, cv2.IMREAD_UNCHANGED)
            if hat_image is not None:
                apply_hat_facemesh(annotated_image, face_landmarks, hat_image, w, h)

        # Glasses placement (can be expanded)
        if "glasses" in accessories and accessories["glasses"] != "none":
            glasses_path = os.path.join(
                asset_dirs["glasses"], accessories["glasses"] + ".png"
            )
            glasses_image = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)
            if glasses_image is not None:
                apply_glasses_facemesh(
                    annotated_image, face_landmarks, glasses_image, w, h
                )

        # Beard placement (can be expanded)
        if "beard" in accessories and accessories["beard"] != "none":
            beard_path = os.path.join(
                asset_dirs["beards"], accessories["beard"] + ".png"
            )
            beard_image = cv2.imread(beard_path, cv2.IMREAD_UNCHANGED)
            if beard_image is not None:
                apply_beard_facemesh(annotated_image, face_landmarks, beard_image, w, h)

    face_mesh.close()
    return annotated_image


def apply_hat_facemesh(
    image, face_landmarks, hat_image, img_width, img_height, hat_scaling_factor=2.5
):
    """
    Places a hat on the forehead using FaceMesh landmarks, accounting for tilt (pitch and yaw).
    """
    hat_scaling_factor = 3.0

    # Key landmarks
    forehead_landmark_ids = [10, 67, 103, 109, 338]
    left_ear_id = 234  # Approximate position for left ear
    right_ear_id = 454  # Approximate position for right ear

    # Forehead landmarks
    forehead_points = [
        (
            int(face_landmarks.landmark[l].x * img_width),
            int(face_landmarks.landmark[l].y * img_height),
        )
        for l in forehead_landmark_ids
    ]
    x_min = min(p[0] for p in forehead_points)
    x_max = max(p[0] for p in forehead_points)
    y_min = min(p[1] for p in forehead_points)
    x_center = (x_min + x_max) // 2

    # Ear positions for yaw calculation
    left_ear = face_landmarks.landmark[left_ear_id]
    right_ear = face_landmarks.landmark[right_ear_id]

    yaw_angle = math.degrees(
        math.atan2(right_ear.y - left_ear.y, right_ear.x - left_ear.x)
    )  # Horizontal tilt

    # Adjust hat size and placement
    hat_width = int((x_max - x_min) * hat_scaling_factor)
    hat_height = int(hat_width * hat_image.shape[0] / hat_image.shape[1])
    resized_hat = cv2.resize(
        hat_image, (hat_width, hat_height), interpolation=cv2.INTER_AREA
    )

    # Rotate hat for yaw alignment
    rotated_hat = rotate_image(resized_hat, -yaw_angle)
    # Adjust the y-coordinate to position the hat slightly lower

    y_adjustment = int(hat_height * 0.10)  # Lower the hat by 25% of its height
    overlay_image(
        image, rotated_hat, x_center - hat_width // 2, y_min - hat_height + y_adjustment
    )


def rotate_image(image, angle):
    """
    Rotates an image by the specified angle around its center.

    Args:
        image (np.ndarray): The image to rotate.
        angle (float): The rotation angle in degrees.

    Returns:
        np.ndarray: The rotated image.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)  # Center of the image
    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    # Perform the rotation
    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return rotated_image


def apply_glasses_facemesh(image, face_landmarks, glasses_image, img_width, img_height):
    # Use eye landmarks for glasses placement
    left_eye_ids = [33, 133]
    right_eye_ids = [362, 263]
    eye_points = [
        (
            int(face_landmarks.landmark[l].x * img_width),
            int(face_landmarks.landmark[l].y * img_height),
        )
        for l in left_eye_ids + right_eye_ids
    ]
    x_min = min(p[0] for p in eye_points)
    x_max = max(p[0] for p in eye_points)
    y_min = min(p[1] for p in eye_points)
    glasses_width = x_max - x_min
    glasses_height = int(
        glasses_width * glasses_image.shape[0] / glasses_image.shape[1]
    )
    resized_glasses = cv2.resize(
        glasses_image, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA
    )
    overlay_image(image, resized_glasses, x_min, y_min)


def apply_beard_facemesh(image, face_landmarks, beard_image, img_width, img_height):
    # Use jawline landmarks for beard placement
    jawline_landmark_ids = list(range(0, 17))
    jawline_points = [
        (
            int(face_landmarks.landmark[l].x * img_width),
            int(face_landmarks.landmark[l].y * img_height),
        )
        for l in jawline_landmark_ids
    ]
    x_min = min(p[0] for p in jawline_points)
    x_max = max(p[0] for p in jawline_points)
    y_max = max(p[1] for p in jawline_points)
    beard_width = x_max - x_min
    beard_height = int(beard_width * beard_image.shape[0] / beard_image.shape[1])
    resized_beard = cv2.resize(
        beard_image, (beard_width, beard_height), interpolation=cv2.INTER_AREA
    )
    overlay_image(image, resized_beard, x_min, y_max - beard_height)


# Main Function
def main():
    load_dotenv()
    API_KEY = os.getenv("OPEN_AI_API_KEY") or ""

    # Initialize OpenAI client
    openai_client = OpenAI_Client(API_KEY, "gpt-4o")

    # Load input image
    input_image_path = "test_images/four-friends.jpg"
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError("Image not found: " + input_image_path)

    # image = preprocess_image(image)

    # Get suggestions from OpenAI
    prompt = open("tag_prompt.txt", "r").read()
    suggestions = openai_client.describe_image_with_retry(input_image_path, prompt)
    print(
        "============================================================================================="
    )
    print(
        "============================================================================================="
    )
    print(f"Extracted Suggestions: {suggestions}")
    print(
        "============================================================================================="
    )
    print(
        "============================================================================================="
    )

    # Asset directories
    asset_dirs = {
        "background": "backgrounds",
        "hats": "hats",
        "glasses": "glasses",
        "beards": "beards",
    }

    # Process suggestions
    for i, suggestion in enumerate(suggestions):
        edited_image = image.copy()

        # Apply background
        if suggestion["Background"] != "none":
            edited_image = apply_background(
                edited_image,
                os.path.join(
                    asset_dirs["background"], suggestion["Background"] + ".jpg"
                ),
            )

        # Apply accessories
        accessories = {
            "hat": suggestion["Hats"],
            "glasses": "none",  # Placeholder if suggestions include glasses
            "beard": "none",  # Placeholder if suggestions include beards
        }
        edited_image = apply_accessories(edited_image, accessories, asset_dirs)

        # Save output
        output_path = f"./output/result_{i+1}.jpg"
        cv2.imwrite(output_path, edited_image)
        cv2.imshow(f"Suggestion {i+1}", edited_image)
        print(f"Saved: {output_path}")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
