# pipeline/accessory_application.py
import os
import cv2
import math
import json
import numpy as np
from pipeline.image_utils import overlay_image, rotate_with_canvas


def load_metadata(json_path):
    """
    Loads accessory metadata from the given JSON file.
    """
    if not os.path.isfile(json_path):
        return {}
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metadata from {json_path}: {e}")
        return {}


class AccessoryPlacer:
    """
    Applies accessories (hats, glasses, masks, and effects) on detected faces.
    """

    def __init__(self, asset_dirs: dict):
        self.asset_dirs = asset_dirs
        self.hat_metadata = load_metadata(
            os.path.join(asset_dirs["hats"], "hats_metadata.json")
        )
        self.glasses_metadata = load_metadata(
            os.path.join(asset_dirs["glasses"], "glasses_metadata.json")
        )
        self.masks_metadata = load_metadata(
            os.path.join(asset_dirs["masks"], "masks_metadata.json")
        )

    def apply_hat(self, image, face_info, hat_name):
        hat_path = os.path.join(self.asset_dirs["hats"], f"{hat_name}.png")
        hat_img = cv2.imread(hat_path, cv2.IMREAD_UNCHANGED)
        if hat_img is None:
            print(f"Hat image not found: {hat_path}")
            return image
        meta = self.hat_metadata.get(hat_name, {})
        left_border = meta.get("left_border", [0, hat_img.shape[0] // 2])
        right_border = meta.get(
            "right_border", [hat_img.shape[1], hat_img.shape[0] // 2]
        )
        hat_anchor = meta.get("brim", [hat_img.shape[1] // 2, hat_img.shape[0]])
        landmarks = face_info["landmarks"]
        left_eye = np.array(landmarks.get("left_eye", [0, 0]), dtype=float)
        right_eye = np.array(landmarks.get("right_eye", [0, 0]), dtype=float)
        eye_center = (left_eye + right_eye) / 2.0
        angle = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        angle_degrees = math.degrees(angle)
        final_angle = -angle_degrees + 180
        up_vector = (math.sin(angle), -math.cos(angle))
        eye_distance = np.linalg.norm(right_eye - left_eye)
        k = 0.6 * eye_distance
        target_anchor = eye_center - k * np.array(up_vector)
        left_border_pt = np.array(left_border, dtype=float)
        right_border_pt = np.array(right_border, dtype=float)
        hat_inner_width = np.linalg.norm(right_border_pt - left_border_pt)
        face_width = 2.2 * eye_distance
        scale = face_width / hat_inner_width
        new_width = int(hat_img.shape[1] * scale)
        new_height = int(hat_img.shape[0] * scale)
        resized_hat = cv2.resize(
            hat_img, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        scaled_anchor = np.array(hat_anchor, dtype=float) * scale
        extra = 150
        rotated_hat = rotate_with_canvas(resized_hat, final_angle, extra=extra)
        padded_w = new_width + 2 * extra
        padded_h = new_height + 2 * extra
        center = (padded_w // 2, padded_h // 2)
        M = cv2.getRotationMatrix2D(center, final_angle, 1.0)
        anchor_in_padded = np.array(
            [scaled_anchor[0] + extra, scaled_anchor[1] + extra, 1.0]
        )
        rotated_anchor = M.dot(anchor_in_padded)
        top_left = target_anchor - rotated_anchor
        result = overlay_image(image, rotated_hat, int(top_left[0]), int(top_left[1]))
        return result

    def apply_glasses(self, image, face_info, glasses_name):
        glasses_path = os.path.join(self.asset_dirs["glasses"], f"{glasses_name}.png")
        glasses_img = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)
        if glasses_img is None:
            print(f"Glasses image not found: {glasses_path}")
            return image
        meta = self.glasses_metadata.get(glasses_name, {})
        anchor_x = meta.get("anchor_x", glasses_img.shape[1] // 2)
        anchor_y = meta.get("anchor_y", glasses_img.shape[0] // 2)
        landmarks = face_info["landmarks"]
        left_eye = landmarks.get("left_eye", [0, 0])
        right_eye = landmarks.get("right_eye", [0, 0])
        eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
        eye_center_y = (left_eye[1] + right_eye[1]) / 2.0
        face_width = abs(right_eye[0] - left_eye[0]) * 2.5
        scale = face_width / glasses_img.shape[1]
        new_width = int(glasses_img.shape[1] * scale)
        new_height = int(glasses_img.shape[0] * scale)
        resized_glasses = cv2.resize(
            glasses_img, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        angle = math.degrees(
            math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        )
        final_angle = -angle + 180
        padded_rotated = rotate_with_canvas(resized_glasses, final_angle, extra=50)
        anchor_x_scaled = anchor_x * scale + 50
        anchor_y_scaled = anchor_y * scale + 50
        x = int(eye_center_x - anchor_x_scaled)
        y = int(eye_center_y - anchor_y_scaled)
        result = overlay_image(image, padded_rotated, x, y)
        return result

    def apply_masks(self, image, face_info, mask_name):
        masks_path = os.path.join(self.asset_dirs["masks"], f"{mask_name}.png")
        mask_img = cv2.imread(masks_path, cv2.IMREAD_UNCHANGED)
        if mask_img is None:
            print(f"Mask image not found: {masks_path}")
            return image
        meta = self.masks_metadata.get(mask_name, {})
        anchor = meta.get("brim", [mask_img.shape[1] // 2, mask_img.shape[0]])
        left_border = meta.get("left_border")
        right_border = meta.get("right_border")
        anchor_x = anchor[0]
        anchor_y = anchor[1]
        landmarks = face_info["landmarks"]
        left_eye = landmarks.get("left_eye", [0, 0])
        right_eye = landmarks.get("right_eye", [0, 0])
        eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
        eye_center_y = (left_eye[1] + right_eye[1]) / 2.0
        face_width = abs(right_eye[0] - left_eye[0]) * 2.5
        mask_inner_width = np.linalg.norm(
            np.array(left_border) - np.array(right_border)
        )
        scale = face_width / mask_inner_width
        new_width = int(mask_img.shape[1] * scale)
        new_height = int(mask_img.shape[0] * scale)
        resized_mask = cv2.resize(
            mask_img, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        angle = math.degrees(
            math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        )
        final_angle = -angle + 180
        padded_rotated = rotate_with_canvas(resized_mask, final_angle, extra=50)
        anchor_x_scaled = anchor_x * scale + 50
        anchor_y_scaled = anchor_y * scale + 50
        x = int(eye_center_x - anchor_x_scaled)
        y = int(eye_center_y - anchor_y_scaled)
        result = overlay_image(image, padded_rotated, x, y)
        return result

    def apply_effect(self, image, effect_name):
        effect_path = os.path.join(self.asset_dirs["effects"], f"{effect_name}.png")
        effect_img = cv2.imread(effect_path, cv2.IMREAD_UNCHANGED)
        if effect_img is None:
            print(f"Effect image not found: {effect_path}")
            return image
        effect_resized = cv2.resize(
            effect_img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA
        )
        result = overlay_image(image, effect_resized, 0, 0)
        return result

    def apply_accessories(self, image, face_info, accessories: dict):
        if accessories.get("hat", "none").lower() != "none":
            image = self.apply_hat(image, face_info, accessories["hat"])
        if accessories.get("glasses", "none").lower() != "none":
            image = self.apply_glasses(image, face_info, accessories["glasses"])
        if accessories.get("masks", "none").lower() != "none":
            image = self.apply_masks(image, face_info, accessories["masks"])
        return image
