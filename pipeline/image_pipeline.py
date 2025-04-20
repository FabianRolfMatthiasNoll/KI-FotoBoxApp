import os
import time
import cv2
import json
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pipeline.image_utils import draw_faces
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

    def process_image(self, image, background_override, effect_override):
        """
        Processes the input image (a numpy array):
        - Generates an asset prompt.
        - In parallel, obtains accessory suggestions from OpenAI and performs face detection.
        - Optionally replaces the background.
        - Overlays accessories on each detected face.
        Returns a list of resulting images (one per suggestion).
        """
        if image is None:
            raise ValueError("Input image is None.")

        overall_start = time.time()
        print("Starting image processing pipeline...")

        # Step 1: Generate the asset prompt.
        t0 = time.time()
        asset_prompt = self.generate_asset_prompt()
        print("Asset prompt generated in {:.3f} seconds.".format(time.time() - t0))
        # Uncomment to print full prompt: print(asset_prompt)

        # Step 2: Save input image temporarily for the OpenAI client.
        t1 = time.time()
        temp_path = "./temp_input.jpg"
        cv2.imwrite(temp_path, image)
        print(
            "Input image saved to temporary file in {:.3f} seconds.".format(
                time.time() - t1
            )
        )

        # Step 3: Run the OpenAI suggestion and face detection in parallel.
        t2 = time.time()
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_suggestions = executor.submit(
                self.openai_client.describe_image_with_retry,
                temp_path,
                prompt=asset_prompt,
            )
            future_faces = executor.submit(self.face_detector.detect_faces, image)
            structured_response = future_suggestions.result()
            faces = future_faces.result()
        print(
            "Received structured suggestions and performed face detection in {:.3f} seconds.".format(
                time.time() - t2
            )
        )
        suggestions = [
            structured_response.suggestion1,
            structured_response.suggestion2,
            structured_response.suggestion3,
        ]
        # Apply override values if provided.
        if background_override is not None and background_override.strip() != "":
            for suggestion in suggestions:
                suggestion.Background = background_override
            print(f"Overriding background with: {background_override}")
        if effect_override is not None and effect_override.strip() != "":
            for suggestion in suggestions:
                suggestion.Effects = effect_override
            print(f"Overriding effect with: {effect_override}")

        if not faces:
            print("No faces detected.")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return [image]

        print("Face detection reported {} faces.".format(len(faces)))

        results = []
        # For each suggestion, create a processed image.
        for idx, suggestion in enumerate(suggestions):
            t_iter = time.time()
            edited_image = image.copy()

            # Step 4: Background Replacement.
            if suggestion.Background.lower() != "none":
                t_bg = time.time()
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
                    t_mask = time.time()
                    mask = self.bg_remover.remove_background(edited_image)
                    print(
                        "Background removal took {:.3f} seconds.".format(
                            time.time() - t_mask
                        )
                    )
                    mask_inv = cv2.bitwise_not(mask)
                    fg = cv2.bitwise_and(edited_image, edited_image, mask=mask)
                    new_bg = cv2.bitwise_and(bg_img, bg_img, mask=mask_inv)
                    edited_image = cv2.add(fg, new_bg)
                    print(
                        "Background replacement completed in {:.3f} seconds.".format(
                            time.time() - t_bg
                        )
                    )

            # Step 5: Assemble accessory specification.
            accessories = {
                "hat": suggestion.Hats,
                "glasses": suggestion.Glasses,
                "effect": suggestion.Effects,
                "masks": suggestion.Masks,
            }

            # Step 6: Apply accessory overlays for each detected face.
            t_accessory = time.time()
            for face_info in faces:
                edited_image = self.accessory_placer.apply_accessories(
                    edited_image, face_info, accessories
                )
            print(
                "Accessory application took {:.3f} seconds.".format(
                    time.time() - t_accessory
                )
            )

            # Step 7: Apply an overall effect overlay if specified.
            if accessories.get("effect", "none").lower() != "none":
                t_effect = time.time()
                edited_image = self.accessory_placer.apply_effect(
                    edited_image, accessories["effect"]
                )
                print(
                    "Effect overlay applied in {:.3f} seconds.".format(
                        time.time() - t_effect
                    )
                )

            # Step 8: Save the processed image for logging/debug.
            output_path = f"./output/result_{idx+1}.jpg"
            cv2.imwrite(output_path, edited_image)
            print(
                "Saved result {} to {} (iteration took {:.3f} seconds).".format(
                    idx + 1, output_path, time.time() - t_iter
                )
            )
            results.append(edited_image)

        # Cleanup temporary file.
        if os.path.exists(temp_path):
            os.remove(temp_path)

        print(
            "Total pipeline processing time: {:.3f} seconds.".format(
                time.time() - overall_start
            )
        )
        return results
