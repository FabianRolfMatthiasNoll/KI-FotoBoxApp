from openai import OpenAI
import base64
import re
import os
from dotenv import load_dotenv


class OpenAI_Client:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def describe_image(self, image_path, prompt) -> str:
        base64_image = self.encode_image(image_path)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that analyzes images and generates strictly formatted tags for a rule-based editing system. Always follow the predefined format without deviation.",
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
        )
        return response.choices[0].message.content or ""

    def process_prompt(self, prompt_text):
        suggestions = []
        suggestion_pattern = re.compile(
            r"Suggestion \d+:(.*?)(?=\nSuggestion \d+:|$)", re.DOTALL
        )
        matches = suggestion_pattern.findall(prompt_text)

        for match in matches:
            suggestion = {
                "Background": "none",
                "Hats": {},
                "Effects": "none",
            }

            try:
                # Extract fields using regex
                bg_match = re.search(r"\[Background\]:\s*([^\n]+)", match)
                if bg_match:
                    suggestion["Background"] = bg_match.group(1).strip()

                hats_match = re.search(r"\[Hats\]:\s*([^\n]+)", match)
                if hats_match:
                    suggestion["Hats"] = hats_match.group(1).strip()

                effects_match = re.search(r"\[Effects\]:\s*([^\n]+)", match)
                if effects_match:
                    suggestion["Effects"] = effects_match.group(1).strip()

                # Handle missing fields
                if not suggestion["Background"] or not suggestion["Effects"]:
                    raise ValueError("Missing required fields in the suggestion.")

                suggestions.append(suggestion)

            except Exception as e:
                print(f"Error processing suggestion: {e}")
                continue

        return suggestions

    def describe_image_with_retry(self, image_path, prompt, retries=3):
        for attempt in range(retries):
            response = self.describe_image(image_path, prompt)
            try:
                # Validate output
                tag_list = self.process_prompt(response)
                if tag_list:  # If valid tags are extracted
                    return tag_list
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")

            print("Retrying...")
        raise ValueError("Failed to get valid response after retries.")


if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.getenv("OPEN_AI_API_KEY") or ""

    client = OpenAI_Client(
        API_KEY,
        "gpt-4o",
    )
    image_path = "./test_images/test.jpg"
    with open("tag_prompt.txt", "r", encoding="utf-8") as file:
        prompt = file.read().strip()

    response = client.describe_image_with_retry(image_path, prompt)

    tag_list = client.process_prompt(response)
    print(tag_list)
