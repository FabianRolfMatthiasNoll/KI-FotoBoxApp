from openai import OpenAI
import base64
import cv2
from models.image_tags import ImageTags, ImageTagsResponse


class OpenAI_Client:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def encode_image(self, image_path):
        # Read the image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        # Compress the image:
        max_width = 640  # Resize if image width is larger than 640 pixels.
        if img.shape[1] > max_width:
            scale = max_width / img.shape[1]
            new_dimensions = (max_width, int(img.shape[0] * scale))
            img = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_AREA)
        # Encode to JPEG with quality 50 (tune quality as needed).
        ret, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if not ret:
            raise ValueError("JPEG encoding failed")
        return base64.b64encode(buf).decode("utf-8")

    def describe_image(self, image_path, prompt) -> ImageTagsResponse:
        base64_image = self.encode_image(image_path)
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that analyzes images for a fun photobox application. "
                        "Generate strictly structured JSON tags for a rule-based editing system. "
                        "Return exactly three separate suggestions. Each suggestion must contain the keys Background, Hats, Glasses, Effects, Masks using only the allowed options. "
                        "You are not allowed to make a suggestion that is completely 'none'. "
                        "If you choose a Mask you can't choose Hats or Glasses for that suggestion and vice versa."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ],
            temperature=0.8,
            response_format=ImageTagsResponse,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            # Fallback with three default suggestions.
            fallback = ImageTagsResponse(
                suggestion1=ImageTags(
                    Background="none",
                    Hats="none",
                    Glasses="none",
                    Effects="none",
                    Masks="none",
                ),
                suggestion2=ImageTags(
                    Background="none",
                    Hats="none",
                    Glasses="none",
                    Effects="none",
                    Masks="none",
                ),
                suggestion3=ImageTags(
                    Background="none",
                    Hats="none",
                    Glasses="none",
                    Effects="none",
                    Masks="none",
                ),
            )
            return fallback
        return parsed

    def describe_image_with_retry(
        self, image_path, prompt="", retries=3
    ) -> ImageTagsResponse:
        for attempt in range(retries):
            try:
                tags = self.describe_image(image_path, prompt)
                if tags:
                    return tags
            except Exception as e:
                print(f"Attempt {attempt+1} failed with error: {e}")
            print("Retrying...")
        raise ValueError("Failed to get valid structured response after retries.")
