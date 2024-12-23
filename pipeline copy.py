import cv2
from dotenv import load_dotenv
import numpy as np
import mediapipe as mp
import math
import torch
from torchvision import transforms
from openai_client import OpenAI_Client
from u2net import U2NET
import os
from openai import OpenAI
import re


# Load U^2-Net model
def load_u2net_model(model_path):
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    net.eval()
    return net


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
    mask = (prediction > 0.5).astype(np.uint8)
    return cv2.resize(
        mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
    )


# Overlay an image with transparency
def overlay_image(background, overlay, x, y):
    h, w = overlay.shape[:2]
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        background[y : y + h, x : x + w, c] = (
            alpha * overlay[:, :, c] + (1 - alpha) * background[y : y + h, x : x + w, c]
        )
    return background


# Apply background to an image
def apply_background(image, background_path, mask):
    background = cv2.imread(background_path)
    if background is None:
        raise ValueError(f"Background not found: {background_path}")
    background = cv2.resize(background, (image.shape[1], image.shape[0]))
    return np.where(mask[..., None] == 1, image, background)


# Apply hat to detected people
def apply_hat(image, bounding_boxes, hat_path):
    hat = cv2.imread(hat_path, cv2.IMREAD_UNCHANGED)
    if hat is None:
        raise ValueError(f"Hat not found: {hat_path}")
    for x, y, w, h in bounding_boxes:
        resized_hat = cv2.resize(hat, (w, int(w * hat.shape[0] / hat.shape[1])))
        overlay_image(image, resized_hat, x, y - resized_hat.shape[0])
    return image


# Apply effects
def apply_effects(image, effect_path):
    effect = cv2.imread(effect_path, cv2.IMREAD_UNCHANGED)
    if effect is None:
        raise ValueError(f"Effect not found: {effect_path}")
    effect_resized = cv2.resize(effect, (image.shape[1], image.shape[0]))
    return overlay_image(image, effect_resized, 0, 0)


# Process each suggestion and generate output images
def process_suggestions(image, suggestions, asset_dirs, yolo_model):
    u2net_model = load_u2net_model("u2net/u2net.pth")
    mask = remove_background(image, u2net_model)
    outputs = []
    for i, suggestion in enumerate(suggestions):
        output = image.copy()
        if suggestion["Background"] != "none":
            output = apply_background(
                output,
                os.path.join(
                    asset_dirs["background"], suggestion["Background"] + ".jpg"
                ),
                mask,
            )
        if suggestion["Hats"]:
            bounding_boxes = detect_people_yolo(output, yolo_model)
            for person, hat in suggestion["Hats"].items():
                if hat != "none":
                    apply_hat(
                        output,
                        bounding_boxes,
                        os.path.join(asset_dirs["hats"], hat + ".png"),
                    )
        if suggestion["Effects"] != "none":
            output = apply_effects(
                output,
                os.path.join(asset_dirs["effects"], suggestion["Effects"] + ".png"),
            )
        outputs.append(output)
    return outputs


# Detect people with YOLOv5
def detect_people_yolo(image, model, conf_threshold=0.5):
    results = model(image)
    detections = results.xyxy[0].cpu().numpy()
    return [
        (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        for x1, y1, x2, y2, conf, cls in detections
        if conf > conf_threshold and int(cls) == 0
    ]


# Main program
def main():
    load_dotenv()
    API_KEY = os.getenv("OPEN_AI_API_KEY") or ""

    # Initialize OpenAI and YOLOv5
    openai_client = OpenAI_Client(API_KEY, "gpt-4o")
    yolo_model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

    # Load input image
    input_image_path = "test_images/workgroup.jpg"
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError("Image not found: " + input_image_path)

    # Describe image and parse suggestions
    prompt = open("tag_prompt.txt", "r").read()
    description = openai_client.describe_image(input_image_path, prompt)
    suggestions = openai_client.process_prompt(description)

    # Process each suggestion
    asset_dirs = {"background": "backgrounds", "hats": "hats", "effects": "effects"}
    edited_images = process_suggestions(image, suggestions, asset_dirs, yolo_model)

    # Save and display results
    for i, edited_image in enumerate(edited_images):
        output_path = f"output/result_{i+1}.jpg"
        cv2.imwrite(output_path, edited_image)
        cv2.imshow(f"Suggestion {i+1}", edited_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
