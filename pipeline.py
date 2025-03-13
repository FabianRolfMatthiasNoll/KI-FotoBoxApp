import json
import math
import os
import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from torchvision import transforms

# Import the RetinaFace detector.
# This example uses the 'retinaface' package (pip install retinaface)
from retinaface import RetinaFace

# Import the pre-trained U²-Net model (assumed available as module u2net)
from u2net import U2NET

# Import the OpenAI client (assumed implemented in openai_client.py)
from openai_client import OpenAI_Client


#########################
# Utility Functions
#########################
def overlay_image(background, overlay, x, y):
    """
    Overlays an RGBA image (overlay) onto a BGR background image at position (x, y).
    The overlay image must have an alpha channel.
    """
    h, w = overlay.shape[:2]
    if overlay.shape[2] < 4:
        raise ValueError("Overlay image must have an alpha channel.")
    alpha = overlay[:, :, 3] / 255.0

    # Calculate overlay region on background
    y1 = max(0, y)
    y2 = min(background.shape[0], y + h)
    x1 = max(0, x)
    x2 = min(background.shape[1], x + w)

    # Determine the corresponding region on the overlay image
    overlay_y1 = max(0, -y)
    overlay_y2 = overlay_y1 + (y2 - y1)
    overlay_x1 = max(0, -x)
    overlay_x2 = overlay_x1 + (x2 - x1)

    # Blend the overlay with the background
    for c in range(3):  # for each color channel
        background[y1:y2, x1:x2, c] = (
            alpha[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
            * overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, c]
            + (1 - alpha[overlay_y1:overlay_y2, overlay_x1:overlay_x2])
            * background[y1:y2, x1:x2, c]
        )
    return background


def rotate_image(image, angle):
    """
    Rotates an image by a given angle (in degrees) around its center.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return rotated


def rotate_with_canvas(image, angle, extra=50):
    """
    Rotates an RGBA image by a given angle around its center, ensuring no corners
    are cut off by first placing it on a larger transparent canvas.

    Parameters:
        image (np.ndarray): RGBA image to rotate (H x W x 4).
        angle (float): Rotation angle in degrees.
        extra (int): Extra padding around the image to prevent clipping.

    Returns:
        np.ndarray: The rotated image on a larger canvas, preserving all corners.
    """
    h, w = image.shape[:2]
    # Create a bigger RGBA canvas (extra on each side)
    canvas = np.zeros((h + 2 * extra, w + 2 * extra, 4), dtype=image.dtype)

    # Place the original image in the center of the canvas
    canvas[extra : extra + h, extra : extra + w, :] = image

    # The new center for rotation is the center of this bigger canvas
    center = ((w + 2 * extra) // 2, (h + 2 * extra) // 2)

    # Perform rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        canvas,
        rotation_matrix,
        (w + 2 * extra, h + 2 * extra),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return rotated


def load_hat_metadata(hat_dir):
    """
    Loads hat metadata from hats_metadata.json in the specified directory.
    Returns a dictionary where keys are hat names and values are the config.
    """
    metadata_path = os.path.join(hat_dir, "hats_metadata.json")
    if not os.path.isfile(metadata_path):
        print(f"No hat metadata found at: {metadata_path}. Using defaults.")
        return {}
    with open(metadata_path, "r") as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            print(f"Error parsing hats_metadata.json: {e}")
            return {}


#########################
# Module: FaceDetector
#########################
class FaceDetector:
    """
    Uses RetinaFace (a pre-trained PyTorch model) to detect multiple faces and extract 5 key landmarks.

    Expected landmarks from RetinaFace (for each face):
      - "left_eye": (x, y)
      - "right_eye": (x, y)
      - "nose": (x, y)
      - "mouth_left": (x, y)
      - "mouth_right": (x, y)
    """

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # The RetinaFace package used here performs detection via a simple API.
        # If you need GPU support for the PyTorch variant, ensure your installation supports it.
        # (The current 'retinaface' package abstracts these details.)

    def detect_faces(self, image):
        """
        Detects faces in the given image.
        Returns a list of dictionaries, each with keys 'bbox' and 'landmarks'.
        """
        # The detect_faces function returns a dict with face_id keys.
        faces_dict = RetinaFace.detect_faces(image)
        faces = []
        for face_id, face_data in faces_dict.items():
            bbox = face_data["facial_area"]  # [x1, y1, x2, y2]
            landmarks = face_data[
                "landmarks"
            ]  # dict with keys: "left_eye", "right_eye", etc.
            faces.append({"bbox": bbox, "landmarks": landmarks})
        return faces


#########################
# Module: BackgroundRemover
#########################
class BackgroundRemover:
    """
    Uses U²‑Net (pre-trained) to perform robust background segmentation.
    The segmentation mask is used for alpha blending with a new background.
    """

    def __init__(self, model_path="u2net/u2net.pth", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = U2NET(3, 1)
        state_dict = torch.load(model_path, map_location=torch.device(self.device))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((320, 320)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def remove_background(self, image, threshold=0.6):
        """
        Returns a binary mask (same width & height as image) where the foreground is 1.
        """
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            d1, *rest = self.model(input_tensor)
            prediction = torch.sigmoid(d1[:, 0, :, :]).squeeze().cpu().numpy()
            print(f"Prediction range: min={prediction.min()}, max={prediction.max()}")
        mask = (prediction > threshold).astype(np.uint8)
        mask_resized = cv2.resize(
            mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        return mask_resized


#########################
# Module: AccessoryPlacer
#########################
class AccessoryPlacer:
    """
    Places accessories (hats and glasses) on a face based on detected landmarks.

    The accessory images must be prepared as described in the guidelines:
      - PNG format with transparent background.
      - Tightly cropped, with a clear anchor point (bottom center for hats, center for glasses).
    """

    def __init__(self, asset_dirs):
        """
        asset_dirs: dictionary with keys 'hats', 'glasses', 'backgrounds' etc.
        """
        self.asset_dirs = asset_dirs
        self.hat_metadata = load_hat_metadata(asset_dirs["hats"])

    def apply_hat(
        self,
        image,
        face_info,
        hat_name,
        default_scale_factor=1.5,
        default_rotation_offset=180,
    ):
        """
        Places a hat accessory on the face using a mix of head pose + metadata-based scaling/offset.

        Parameters:
            image (np.ndarray): Original BGR image.
            face_info (dict): Dict with keys 'bbox' and 'landmarks' (x1,y1,x2,y2 plus 5+2D points).
            hat_name (str): Name of the hat asset (without extension).
            default_scale_factor (float): fallback if metadata not found
            default_rotation_offset (float): fallback rotation offset

        Returns:
            np.ndarray: The image with the hat overlaid.
        """
        # Load the hat asset
        hat_path = os.path.join(self.asset_dirs["hats"], hat_name + ".png")
        hat_img = cv2.imread(hat_path, cv2.IMREAD_UNCHANGED)
        if hat_img is None:
            print(f"Hat image not found: {hat_path}")
            return image

        # Read metadata or fall back
        meta = self.hat_metadata.get(hat_name, {})
        scale_mode = meta.get("scale_mode", "width")  # "width" or "height"
        scale_factor = meta.get("scale_factor", default_scale_factor)
        rotation_offset = meta.get("rotation_offset", default_rotation_offset)
        anchor_offset = meta.get("anchor_offset", 0)

        # Retrieve bounding box & landmarks
        bbox = face_info["bbox"]  # [x1, y1, x2, y2]
        landmarks = face_info["landmarks"]

        left_eye = landmarks["left_eye"]
        right_eye = landmarks["right_eye"]

        x1, y1, x2, y2 = bbox
        face_width = x2 - x1
        face_height = y2 - y1

        # Compute angle from eye line
        angle = math.degrees(
            math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        )
        # final angle for the hat
        final_angle = -angle + rotation_offset

        # Scale the hat
        if scale_mode == "width":
            # Scale by face width
            hat_width = int(face_width * scale_factor)
            hat_height = int(hat_width * hat_img.shape[0] / hat_img.shape[1])
        else:
            # scale_mode == "height"
            hat_height = int(face_height * scale_factor)
            hat_width = int(hat_height * (hat_img.shape[1] / hat_img.shape[0]))

        resized_hat = cv2.resize(
            hat_img, (hat_width, hat_height), interpolation=cv2.INTER_AREA
        )

        # Rotate the hat
        padding_value = 50
        rotated_hat = rotate_with_canvas(resized_hat, final_angle, extra=padding_value)

        # 7) Determine anchor point for the brim
        # We'll place the brim at y1 (top of face bounding box) minus anchor_offset
        # so the brim sits slightly above the top. Adjust as needed.
        brim_y = y1 - anchor_offset
        # horizontally, we center it on the face
        face_center_x = int((left_eye[0] + right_eye[0]) / 2.0)

        angle_radians = math.radians(angle)
        sin_term = math.sin(angle_radians)
        delta_y = right_eye[1] - left_eye[1]

        # Weighted combination
        x_offset = int(0.1 * hat_width * sin_term + 0.15 * delta_y)

        # 8) Place the hat so its bottom aligns with brim_y
        x = face_center_x - (rotated_hat.shape[1] // 2) - 3 * x_offset
        y = brim_y - rotated_hat.shape[0] + padding_value

        # 9) Overlay
        result = overlay_image(image, rotated_hat, x, y)
        return result

    def apply_glasses(self, image, landmarks, glasses_name):
        """
        Places glasses on the face.
        The glasses are scaled based on the inter-eye distance and rotated to match head tilt.
        """
        glasses_path = os.path.join(self.asset_dirs["glasses"], glasses_name + ".png")
        glasses_img = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)
        if glasses_img is None:
            print(f"Glasses image not found: {glasses_path}")
            return image

        left_eye = landmarks["left_eye"]
        right_eye = landmarks["right_eye"]
        center_x = int((left_eye[0] + right_eye[0]) / 2)
        center_y = int((left_eye[1] + right_eye[1]) / 2)
        eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))

        # Scale glasses slightly larger than the distance between eyes
        glasses_width = int(eye_distance * 1.5)
        glasses_height = int(
            glasses_width * glasses_img.shape[0] / glasses_img.shape[1]
        )
        resized_glasses = cv2.resize(
            glasses_img, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA
        )
        angle = math.degrees(
            math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        )
        rotated_glasses = rotate_image(resized_glasses, -angle)

        x = center_x - glasses_width // 2
        y = center_y - glasses_height // 2
        result = overlay_image(image, rotated_glasses, x, y)
        return result

    def apply_accessories(self, image, face_info, accessories):
        """
        Applies the specified accessories to the image for one detected face.

        accessories: dict with keys such as "hat", "glasses" (values are asset names or "none")
        """
        landmarks = face_info["landmarks"]
        if accessories.get("hat", "none") != "none":
            image = self.apply_hat(image, face_info, accessories["hat"])
        if accessories.get("glasses", "none") != "none":
            image = self.apply_glasses(image, landmarks, accessories["glasses"])
        return image


#########################
# Module: Pipeline Orchestrator
#########################
class ImagePipeline:
    """
    Main pipeline that combines face detection, background removal, accessory placement,
    and optional style suggestions (via OpenAI) into a single processing flow.
    """

    def __init__(self, asset_dirs, u2net_model_path, openai_api_key, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.face_detector = FaceDetector(device=self.device)
        self.bg_remover = BackgroundRemover(
            model_path=u2net_model_path, device=self.device
        )
        self.accessory_placer = AccessoryPlacer(asset_dirs)
        self.openai_client = OpenAI_Client(openai_api_key, "gpt-4o")

    def process_image(self, image_path):
        """
        Processes the image:
          - Detects faces
          - Optionally replaces background
          - Applies accessories based on suggestions from OpenAI

        Returns a list of resulting images (one per suggestion).
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found: " + image_path)

        # Get suggestions from OpenAI (expects a tag_prompt.txt file to exist)
        with open("tag_prompt.txt", "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        suggestions = self.openai_client.describe_image_with_retry(image_path, prompt)
        if not suggestions:
            print("No valid suggestions received; using default accessories.")
            suggestions = [{"Background": "none", "Hats": "none", "Glasses": "none"}]

        # Detect faces in the image using RetinaFace
        faces = self.face_detector.detect_faces(image)
        if not faces:
            print("No faces detected.")
            return [image]

        results = []
        for idx, suggestion in enumerate(suggestions):
            edited_image = image.copy()

            # --- Background Replacement ---
            if suggestion.get("Background", "none") != "none":
                bg_path = os.path.join(
                    self.accessory_placer.asset_dirs["background"],
                    suggestion["Background"] + ".jpg",
                )
                bg_img = cv2.imread(bg_path)
                if bg_img is None:
                    print(
                        f"Background {bg_path} not found; skipping background replacement."
                    )
                else:
                    bg_img = cv2.resize(
                        bg_img, (edited_image.shape[1], edited_image.shape[0])
                    )
                    mask = self.bg_remover.remove_background(edited_image)
                    mask_inv = cv2.bitwise_not(mask)
                    fg = cv2.bitwise_and(edited_image, edited_image, mask=mask)
                    new_bg = cv2.bitwise_and(bg_img, bg_img, mask=mask_inv)
                    edited_image = cv2.add(fg, new_bg)

            # --- Accessory Placement ---
            accessories = {
                "hat": suggestion.get("Hats", "none"),
                "glasses": suggestion.get("Glasses", "none"),
            }
            for face_info in faces:
                edited_image = self.accessory_placer.apply_accessories(
                    edited_image, face_info, accessories
                )
            # Save and store result
            output_path = f"./output/result_{idx+1}.jpg"
            cv2.imwrite(output_path, edited_image)
            print(f"Saved result to {output_path}")
            results.append(edited_image)
        return results


#########################
# Main Execution
#########################
def main():
    load_dotenv()
    API_KEY = os.getenv("OPEN_AI_API_KEY") or ""

    # Define asset directories (adjust paths as needed)
    asset_dirs = {
        "background": "backgrounds",
        "hats": "hats",
        # "glasses": "glasses",
        # Optionally add more accessory types here (e.g., "beards": "beards")
    }
    # Ensure output directory exists
    os.makedirs("./output", exist_ok=True)

    # Initialize the pipeline (ensure paths to models and assets are correct)
    pipeline = ImagePipeline(
        asset_dirs=asset_dirs,
        u2net_model_path="u2net/u2net.pth",
        openai_api_key=API_KEY,
    )

    # Path to the input image (from backend)
    input_image_path = "test_images/four-friends.jpg"
    results = pipeline.process_image(input_image_path)

    # Optionally display results (close windows by pressing any key)
    for idx, res in enumerate(results):
        cv2.imshow(f"Result {idx+1}", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
