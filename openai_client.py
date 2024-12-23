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
                    "content": "You are an assistant to describe an image in detail and generate tags for a fotobox so that the images can be edited based on the content in the following format:"
                    + prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
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
        )
        return response.choices[0].message.content or ""

    def process_prompt(self, prompt_text):
        """
        Processes the structured prompt text and extracts keywords into a usable format.

        Args:
            prompt_text (str): The input prompt containing suggestions and tags.

        Returns:
            list[dict]: A list of suggestions, each represented as a dictionary with keys:
                        'Background', 'Hats', 'Effects', and 'Comments'.
        """
        suggestions = []
        # Debugging output
        print("Processing Prompt:\n", prompt_text)

        # Regular expression to split suggestions
        suggestion_pattern = re.compile(
            r"Suggestion \d+:(.*?)(?=\nSuggestion \d+:|$)", re.DOTALL
        )
        matches = suggestion_pattern.findall(prompt_text)

        for match in matches:
            suggestion = {
                "Background": "none",
                "Hats": {},
                "Effects": "none",
                "Comments": "",
            }

            # Extract Background
            bg_match = re.search(r"\[Background\]:\s*([^\n]+)", match)
            if bg_match:
                suggestion["Background"] = bg_match.group(1).strip()

            # Extract Hats
            hats_match = re.search(r"\[Hats\]:\s*\[([^\]]+)\]", match)
            if hats_match:
                hats_text = hats_match.group(1)
                person_matches = re.findall(r"Person (\d+):\s*([\w_]+)", hats_text)
                suggestion["Hats"] = {
                    f"Person {person_id}": hat for person_id, hat in person_matches
                }

            # Extract Effects
            effects_match = re.search(r"\[Effects\]:\s*([^\n]+)", match)
            if effects_match:
                suggestion["Effects"] = effects_match.group(1).strip()

            # Extract Comments
            comments_match = re.search(
                r"\[Comments\]:\s*(.*?)(?=\n|$)", match, re.DOTALL
            )
            if comments_match:
                suggestion["Comments"] = comments_match.group(1).strip()

            # Fill missing Hats with "none"
            if suggestion["Hats"]:
                max_people = max(int(pid.split()[-1]) for pid in suggestion["Hats"])
                for person_id in range(1, max_people + 1):
                    person_key = f"Person {person_id}"
                    if person_key not in suggestion["Hats"]:
                        suggestion["Hats"][person_key] = "none"

            suggestions.append(suggestion)

        # Debugging output
        print("Extracted Suggestions:", suggestions)
        return suggestions


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

    response = client.describe_image(image_path, prompt)
    print("Response: " + response)
    print("===============================")
    tag_list = client.process_prompt(response)
    print(tag_list)
