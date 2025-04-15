import os
import cv2
import json
from pipeline.image_utils import (
    draw_faces,
)
from pipeline import (
    global_vars,
)  # Contains: face_detector, bg_remover, accessory_placer, openai_client


def load_all_assets(asset_dirs):
    """
    Loads asset options for all categories.
    For categories with metadata (hats, glasses, masks), keys are loaded from metadata JSON.
    For backgrounds and effects, the file names (without extension) are scanned.
    Always adds "none" if not present.
    """
    all_assets = {}
    # For file-based assets (backgrounds, effects):
    for category in ["backgrounds", "effects"]:
        directory = asset_dirs.get(category, "")
        if not os.path.isdir(directory):
            all_assets[category] = ["none"]
        else:
            files = [
                os.path.splitext(f)[0]
                for f in os.listdir(directory)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            if "none" not in files:
                files.append("none")
            all_assets[category] = sorted(files)
    # For metadata-based assets (hats, glasses, masks):
    for category in ["hats", "glasses", "masks"]:
        meta_file = os.path.join(
            asset_dirs.get(category, ""), f"{category}_metadata.json"
        )
        if os.path.isfile(meta_file):
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                keys = list(meta.keys())
                if "none" not in keys:
                    keys.append("none")
                all_assets[category] = sorted(keys)
            except Exception as e:
                print(f"Error loading {category} metadata: {e}")
                all_assets[category] = ["none"]
        else:
            all_assets[category] = ["none"]
    return all_assets


class ImagePipeline:
    """
    Main pipeline that combines face detection, background removal,
    accessory placement, and style suggestions (via OpenAI) into a single process.
    """

    def __init__(self, asset_dirs, device=None):
        # Although models were loaded globally in app/startup, we still need our asset settings.
        self.device = device or (
            "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        )
        # Retrieve the globally loaded models.
        self.face_detector = global_vars.face_detector
        self.bg_remover = global_vars.bg_remover
        self.accessory_placer = global_vars.accessory_placer
        self.openai_client = global_vars.openai_client

        self.asset_dirs = asset_dirs
        self.asset_options = load_all_assets(asset_dirs)

    def generate_asset_prompt(self):
        """
        Generates a text prompt listing available asset options for each category.
        """
        backgrounds = ", ".join(self.asset_options.get("backgrounds", []))
        hats = ", ".join(self.asset_options.get("hats", []))
        glasses = ", ".join(self.asset_options.get("glasses", []))
        effects = ", ".join(self.asset_options.get("effects", []))
        masks = ", ".join(self.asset_options.get("masks", []))
        prompt = (
            "Analyze the uploaded image and generate structured tags for a rule-based editing system. "
            "Return exactly three separate suggestions. Use only the following options:\n"
            f"[Background]: {backgrounds}\n"
            f"[Hats]: {hats}\n"
            f"[Glasses]: {glasses}\n"
            f"[Effects]: {effects}\n"
            f"[Masks]: {masks}\n"
            "If the image shows heart gestures, for example, use 'heart' for Effects and 'glasses_heart' for Glasses. "
            "Try to make funny combinations. Things like a space background and astronaut masks could be combined and triggered by waving the arms as an example."
        )
        return prompt

    def process_image(self, image):
        """
        Processes the input image (a numpy array):
          - Detects faces.
          - Generates a prompt using asset options.
          - Obtains accessory suggestions from OpenAI.
          - Optionally replaces the background.
          - Overlays accessories on each detected face.
        Returns a list of resulting images (one per suggestion).
        """
        if image is None:
            raise ValueError("Input image is None.")

        # Generate the asset prompt based on available assets.
        asset_prompt = self.generate_asset_prompt()
        print("Generated asset prompt...")
        # print(asset_prompt)

        # Save the image temporarily to supply a file path for the OpenAI client.
        temp_path = "./temp_input.jpg"
        cv2.imwrite(temp_path, image)

        # Get structured suggestions from the OpenAI client.
        structured_response = self.openai_client.describe_image_with_retry(
            temp_path, prompt=asset_prompt
        )
        suggestions = [
            structured_response.suggestion1,
            structured_response.suggestion2,
            structured_response.suggestion3,
        ]
        print("Received structured suggestions...")
        # print(suggestions)

        # Perform face detection.
        faces = self.face_detector.detect_faces(image)
        if not faces:
            print("No faces detected.")
            return [image]

        # Optionally save a debug image with face bounding boxes and landmarks.
        face_debug_image = draw_faces(image, faces)
        cv2.imwrite("./output/faces_debug.jpg", face_debug_image)
        print("Saved face landmark debug image to ./output/faces_debug.jpg")

        results = []
        for idx, suggestion in enumerate(suggestions):
            edited_image = image.copy()

            # If a background is suggested (not "none"), perform background replacement.
            if suggestion.Background.lower() != "none":
                bg_path = os.path.join(
                    self.accessory_placer.asset_dirs["backgrounds"],
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

            # Assemble the accessory specification.
            accessories = {
                "hat": suggestion.Hats,
                "glasses": suggestion.Glasses,
                "effect": suggestion.Effects,
                "masks": suggestion.Masks,
            }

            # Apply accessory overlays for each detected face.
            for face_info in faces:
                edited_image = self.accessory_placer.apply_accessories(
                    edited_image, face_info, accessories
                )

            # Apply a full-image effect overlay if specified.
            if accessories.get("effect", "none").lower() != "none":
                edited_image = self.accessory_placer.apply_effect(
                    edited_image, accessories["effect"]
                )

            output_path = f"./output/result_{idx+1}.jpg"
            cv2.imwrite(output_path, edited_image)
            print(f"Saved result to {output_path}")
            results.append(edited_image)

        # Cleanup temporary file.
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return results
