from openai import OpenAI
import base64
import json
from models.image_tags import ImageTags, ImageTagsResponse
from dotenv import load_dotenv

prompt = (
    "Analyze the uploaded image and generate structured tags for a rule-based editing system. "
    "Return exactly three separate suggestions. Use only the following options:\n"
    "[Background]: beach, sunset, space, none\n"
    "[Hats]: party_hat, crown, cowboy_hat, none\n"
    "[Glasses]: glasses_clown_nose, glasses_googley_eyes, glasses_nose_mustache, glasses_heart, none\n"
    "[Effects]: confetti, sparkles, spotlight, heart, none\n"
    "If the image shows heart gestures, for example, use 'heart' for Effects and 'glasses_heart' for Glasses."
)


class OpenAI_Client:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def describe_image(self, image_path) -> ImageTagsResponse:
        base64_image = self.encode_image(image_path)
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that analyzes images for a fun photobox application. "
                        "Generate strictly structured JSON tags for a rule-based editing system. "
                        "Return exactly three separate suggestions as a JSON object with keys suggestion1, suggestion2, and suggestion3. "
                        "Each suggestion must contain the keys Background, Hats, Glasses, and Effects, using only the allowed options."
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
            temperature=0.5,
            response_format=ImageTagsResponse,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            # Fallback with three default suggestions.
            fallback = ImageTagsResponse(
                suggestion1=ImageTags(
                    Background="none", Hats="none", Glasses="none", Effects="none"
                ),
                suggestion2=ImageTags(
                    Background="none", Hats="none", Glasses="none", Effects="none"
                ),
                suggestion3=ImageTags(
                    Background="none", Hats="none", Glasses="none", Effects="none"
                ),
            )
            return fallback
        return parsed

    def describe_image_with_retry(self, image_path, retries=3) -> ImageTagsResponse:
        for attempt in range(retries):
            try:
                tags = self.describe_image(image_path)
                if tags:
                    return tags
            except Exception as e:
                print(f"Attempt {attempt+1} failed with error: {e}")
            print("Retrying...")
        raise ValueError("Failed to get valid structured response after retries.")
