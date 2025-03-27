import json
import math
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from dotenv import load_dotenv
from torchvision import transforms

# Import the RetinaFace detector.
# This example uses the 'retinaface' package (pip install retinaface)
from retinaface import RetinaFace


from MODNet.src.models.modnet import MODNet

# Import the OpenAI client (assumed implemented in openai_client.py)
from vision.openai_client import OpenAI_Client


#########################
# Utility Functions
#########################
def overlay_image(background: np.ndarray, overlay, x, y):
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


def rotate_with_canvas(image: np.ndarray, angle: float, extra=50):
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


def load_glasses_metadata(glasses_dir):
    metadata_path = os.path.join(glasses_dir, "glasses_metadata.json")
    if not os.path.isfile(metadata_path):
        print(f"No glasses metadata found at: {metadata_path}. Using defaults.")
        return {}
    with open(metadata_path, "r") as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            print(f"Error parsing glasses_metadata.json: {e}")
            return {}


def draw_faces(image, faces, draw_asset_points=False):
    """
    Draws bounding boxes and facial landmarks on a copy of the image.
    Optionally draws the asset anchor points used for accessory placement.
    """
    output = image.copy()
    for face in faces:
        bbox = face["bbox"]  # [x1, y1, x2, y2]
        landmarks = face["landmarks"]

        # Bounding box (green)
        cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Landmarks (red)
        for point in landmarks.values():
            cv2.circle(output, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)

        if draw_asset_points:
            # === Compute hat anchor debug points ===
            left_eye = np.array(landmarks["left_eye"], dtype=float)
            right_eye = np.array(landmarks["right_eye"], dtype=float)

            # Eye center
            eye_center = (left_eye + right_eye) / 2.0
            cv2.circle(
                output, tuple(eye_center.astype(int)), 4, (255, 255, 0), -1
            )  # yellow

            # Up vector from eye line
            angle = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
            up_vector = (math.sin(angle), -math.cos(angle))
            eye_distance = np.linalg.norm(right_eye - left_eye)

            k = 0.8 * eye_distance
            target = eye_center - k * np.array(up_vector)
            cv2.circle(
                output, tuple(target.astype(int)), 5, (255, 0, 255), -1
            )  # purple

            # Label them if needed (optional)
            # cv2.putText(output, "target", tuple(target.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    return output


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
    Verwendet MODNet (vortrainiert) zur Hintergrundsegmentierung.
    Das Modell gibt ein Alpha-Matte zurück, das für den Alpha-Blending verwendet wird.
    """

    def __init__(self, model_path="modnet/modnet.ckpt", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MODNet(backbone_pretrained=False)

        self.model = nn.DataParallel(self.model)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            weights = torch.load(model_path)
        else:
            weights = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(weights)
        self.model.eval()

    def remove_background(self, image):
        original_size = (image.shape[1], image.shape[0])
        ref_size = 512

        # Konvertiere BGR (cv2) zu RGB und dann in ein PIL-Image
        im_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Definiere den Transformations-Workflow wie im Original-Repo
        im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Wandle das Bild in einen Tensor um und füge eine Batch-Dimension hinzu
        im_tensor = im_transform(im_pil).unsqueeze(0)  # type: ignore # Form: (1, 3, H, W)
        _, _, im_h, im_w = im_tensor.shape

        # Berechne neue Bilddimensionen basierend auf dem Referenzwert
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            else:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        # Passe Höhe und Breite so an, dass sie durch 32 teilbar sind
        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32

        # Resample das Bild auf die neuen Dimensionen
        im_tensor = F.interpolate(im_tensor, size=(im_rh, im_rw), mode="area")

        with torch.no_grad():
            input_tensor = im_tensor.to(self.device)
            # Inferenz – beachte das zusätzliche Flag (hier True) für den Inferenzmodus
            _, _, matte = self.model(input_tensor, True)
            # Skaliere die Matte zurück auf die Originalgröße
            matte = F.interpolate(
                matte, size=(original_size[1], original_size[0]), mode="area"
            )
            matte = matte[0][0].data.cpu().numpy()

        # Wandle das Ergebnis in eine Maske um
        mask = (matte * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite("output/mask.png", binary_mask)
        return binary_mask


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
        self.glasses_metadata = load_glasses_metadata(asset_dirs["glasses"])

    def apply_hat(self, image: np.ndarray, face_info: dict, hat_name: str):
        """
        Places a hat on the face using the key metadata points:
        - left_border and right_border: define the inner hat width.
        - brim: the anchor point that will be aligned above the eyes.

        The hat is scaled so that the inner width of the hat matches the face width,
        rotated based on the eye-line angle, and positioned such that the scaled brim point
        lands above the eye center by an offset proportional to the eye distance.
        """
        # Load the hat asset.
        hat_path = os.path.join(self.asset_dirs["hats"], hat_name + ".png")
        hat_img = cv2.imread(hat_path, cv2.IMREAD_UNCHANGED)
        if hat_img is None:
            print(f"Hat image not found: {hat_path}")
            return image

        # Get the key metadata points.
        meta = self.hat_metadata.get(hat_name, {})
        try:
            left_border = np.array(meta["left_border"], dtype=float)
            right_border = np.array(meta["right_border"], dtype=float)
            brim = np.array(meta["brim"], dtype=float)
        except KeyError:
            print(
                f"Metadata for {hat_name} is missing required points. Using defaults."
            )
            left_border = np.array([0, hat_img.shape[0] // 2], dtype=float)
            right_border = np.array(
                [hat_img.shape[1], hat_img.shape[0] // 2], dtype=float
            )
            brim = np.array([hat_img.shape[1] / 2, hat_img.shape[0]], dtype=float)

        # Compute the hat's inner width (distance between the left and right border).
        hat_inner_width = np.linalg.norm(right_border - left_border)

        # Retrieve face info and compute face width.
        bbox = face_info["bbox"]  # [x1, y1, x2, y2]
        landmarks = face_info["landmarks"]
        left_eye = np.array(landmarks["left_eye"], dtype=float)
        right_eye = np.array(landmarks["right_eye"], dtype=float)
        face_width = bbox[2] - bbox[0]

        # Scale hat so its inner width matches the face width.
        # Optionally multiply by a scale factor from metadata (default 1.0).
        scale_factor = meta.get("scale_factor", 1.0)
        scale = (face_width / hat_inner_width) * scale_factor

        # Resize the hat asset.
        new_w = int(hat_img.shape[1] * scale)
        new_h = int(hat_img.shape[0] * scale)
        resized_hat = cv2.resize(hat_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Compute the scaled position of the hat's anchor (the brim point).
        brim_scaled = brim * scale

        # Determine face rotation based on eye positions.
        angle = math.degrees(
            math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        )
        rotation_offset = meta.get("rotation_offset", 0)
        final_angle = -angle + rotation_offset + 180

        # Rotate the hat with a canvas to prevent clipping.
        extra_padding = 50
        rotated_hat = rotate_with_canvas(resized_hat, final_angle, extra=extra_padding)

        # After rotation, the new anchor (brim point) is offset by the extra padding.
        hat_anchor_x = brim_scaled[0] + extra_padding
        hat_anchor_y = brim_scaled[1] + extra_padding

        # Compute the eye center.
        eye_center = (
            (left_eye[0] + right_eye[0]) / 2.0,
            (left_eye[1] + right_eye[1]) / 2.0,
        )
        eye_distance = np.linalg.norm(right_eye - left_eye)

        # Compute an upward direction based on face rotation.
        a = math.radians(angle)
        up_vector = (math.sin(a), -math.cos(a))

        # Adjust upward from the eye center; tweak 'k' to position the hat on the forehead.
        k = 0.8 * eye_distance  # You can experiment with this multiplier.
        target_x = eye_center[0] - k * up_vector[0]
        target_y = eye_center[1] - k * up_vector[1]

        # Determine overlay position so that the hat's anchor aligns with the target.
        overlay_x = int(target_x - hat_anchor_x)
        overlay_y = int(target_y - hat_anchor_y)

        result = overlay_image(image, rotated_hat, overlay_x, overlay_y)
        return result

    def apply_glasses(self, image, face_info, glasses_name):
        """
        Places glasses on the face using metadata for alignment.
        This version supports a new scale mode "head_width" which scales glasses
        nearly as wide as the face.
        """
        # Load the glasses asset
        glasses_path = os.path.join(self.asset_dirs["glasses"], glasses_name + ".png")
        glasses_img = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)
        if glasses_img is None:
            print(f"Glasses image not found: {glasses_path}")
            return image

        # Retrieve metadata
        meta = self.glasses_metadata.get(glasses_name, {})
        anchor_x = meta.get("anchor_x", glasses_img.shape[1] // 2)
        anchor_y = meta.get("anchor_y", glasses_img.shape[0] // 2)
        scale_factor = meta.get("scale_factor")
        rotation_offset = meta.get("rotation_offset", 0)
        offset_x = meta.get("offset_x", 0)
        offset_y = meta.get("offset_y", 0)

        # Retrieve landmarks; we also need face bbox to estimate head width.
        face_info_bbox = face_info.get("bbox", None)
        landmarks = face_info["landmarks"]
        left_eye = landmarks["left_eye"]
        right_eye = landmarks["right_eye"]

        # Eye midpoint: where we want the glasses' bridge to align.
        eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
        eye_center_y = (left_eye[1] + right_eye[1]) / 2.0

        # Determine base scale
        if face_info_bbox:
            x1, _, x2, _ = face_info_bbox
            face_width = x2 - x1
        else:
            # Fallback: estimate head width as 2.5 * eye distance
            eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
            face_width = 2.5 * eye_distance
        scale = (face_width / glasses_img.shape[1]) * scale_factor

        # Resize glasses based on computed scale
        new_width = int(glasses_img.shape[1] * scale)
        new_height = int(glasses_img.shape[0] * scale)
        resized_glasses = cv2.resize(
            glasses_img, (new_width, new_height), interpolation=cv2.INTER_AREA
        )

        # Compute rotation from the eye line
        angle = math.degrees(
            math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        )
        # Adjust angle by rotation_offset; add 180 if needed for proper orientation
        final_angle = -angle + rotation_offset + 180

        # Use rotate_with_canvas to avoid cutoffs
        padded_rotated = rotate_with_canvas(resized_glasses, final_angle, extra=50)

        # Compute the anchor in the rotated image.
        # The anchor in the original resized image is (anchor_x * scale, anchor_y * scale).
        # After padding, add the extra padding value (50).
        anchor_x_scaled = anchor_x * scale + 50
        anchor_y_scaled = anchor_y * scale + 50

        # Place the glasses so that the anchor lands at the eye center,
        # with additional offsets if defined.
        x = int(eye_center_x - anchor_x_scaled + offset_x)
        y = int(eye_center_y - anchor_y_scaled + offset_y)

        result = overlay_image(image, padded_rotated, x, y)
        return result

    def apply_effect(self, image: np.ndarray, effect_name: str):
        """
        Overlays an effect on the image. The effect asset is expected to be a PNG with transparency.
        It is resized to match the input image dimensions and then overlaid from the top-left corner.
        """
        effect_path = os.path.join(self.asset_dirs["effects"], effect_name + ".png")
        effect_img = cv2.imread(effect_path, cv2.IMREAD_UNCHANGED)
        if effect_img is None:
            print(f"Effect image not found: {effect_path}")
            return image

        # Resize the effect to match the full image size.
        effect_resized = cv2.resize(
            effect_img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA
        )

        # Overlay the effect onto the image. (Position (0,0) is used here.)
        result = overlay_image(image, effect_resized, 0, 0)
        return result

    def apply_accessories(self, image, face_info, accessories):
        """
        Applies the specified accessories to the image for one detected face.

        accessories: dict with keys such as "hat", "glasses" (values are asset names or "none")
        """
        if accessories.get("hat", "none") != "none":
            image = self.apply_hat(image, face_info, accessories["hat"])
        if accessories.get("glasses", "none") != "none":
            image = self.apply_glasses(image, face_info, accessories["glasses"])
        return image


#########################
# Module: Pipeline Orchestrator
#########################
class ImagePipeline:
    """
    Main pipeline that combines face detection, background removal, accessory placement,
    and optional style suggestions (via OpenAI) into a single processing flow.
    """

    def __init__(
        self, asset_dirs, segmentation_model_path, openai_api_key, device=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.face_detector = FaceDetector(device=self.device)
        self.bg_remover = BackgroundRemover(
            model_path=segmentation_model_path, device=self.device
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

        # To draw the calculated placement points of the assets on the debug image
        DRAW_ASSET_DEBUG_POINTS = True

        structured_response = self.openai_client.describe_image_with_retry(image_path)
        suggestions = [
            structured_response.suggestion1,
            structured_response.suggestion2,
            structured_response.suggestion3,
        ]
        print("Received structured suggestions:", suggestions)

        faces = self.face_detector.detect_faces(image)
        if not faces:
            print("No faces detected.")
            return [image]
        
        # Save face-only diagnostic image
        face_debug_image = draw_faces(image, faces, draw_asset_points=DRAW_ASSET_DEBUG_POINTS)
        cv2.imwrite("./output/faces_debug.jpg", face_debug_image)
        print("Saved face landmark debug image to ./output/faces_debug.jpg")

        results = []
        for idx, suggestion in enumerate(suggestions):
            edited_image = image.copy()

            if suggestion.Background != "none":
                bg_path = os.path.join(
                    self.accessory_placer.asset_dirs["background"],
                    suggestion.Background + ".jpg",
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

            accessories = {
                "hat": suggestion.Hats,
                "glasses": suggestion.Glasses,
                "effect": suggestion.Effects,
            }

            for face_info in faces:
                edited_image = self.accessory_placer.apply_accessories(
                    edited_image, face_info, accessories
                )

            if accessories.get("effect", "none") != "none":
                edited_image = self.accessory_placer.apply_effect(
                    edited_image, accessories["effect"]
                )

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

    # Define asset directories (adjust paths as needed). Key - Path
    asset_dirs = {
        "background": "assets/backgrounds",
        "hats": "assets/hats",
        "glasses": "assets/glasses",
        "effects": "assets/effects",
    }
    # Ensure output directory exists
    os.makedirs("./output", exist_ok=True)

    # Initialize the pipeline (ensure paths to models and assets are correct)
    pipeline = ImagePipeline(
        asset_dirs=asset_dirs,
        segmentation_model_path="checkpoints/modnet_photographic_portrait_matting.ckpt",
        openai_api_key=API_KEY,
    )

    # Path to the input image (from backend)
    input_image_path = "assets/test_images/lovers.jpg"
    pipeline.process_image(input_image_path)


if __name__ == "__main__":
    main()
