import ollama
import base64
import os


class OllamaLLaVAClient:
    def __init__(self, model_name="llava:34b"):
        """
        Initialize the Ollama client for the LLaVA model.
        Args:
            model_name (str): The name of the LLaVA model to use.
        """
        self.model_name = model_name

    def read_prompt_from_file(self, prompt_file):
        """
        Read the prompt from a text file.
        Args:
            prompt_file (str): Path to the prompt file.
        Returns:
            str: The content of the prompt file.
        """
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        with open(prompt_file, "r", encoding="utf-8") as file:
            return file.read().strip()

    def query_model(self, prompt_file, image_path=None, stream=True):
        """
        Query the LLaVA model with a prompt file and optional image.
        Args:
            prompt_file (str): Path to the text file containing the prompt.
            image_path (str): Path to the image file (optional).
            stream (bool): Whether to use streaming responses.
        Returns:
            str: The generated response from the model.
        """
        try:
            # Read the prompt from the file
            prompt = self.read_prompt_from_file(prompt_file)
            print(prompt)
            # Prepare the arguments
            args = {"model": self.model_name, "prompt": prompt, "stream": stream}

            # Add image if provided
            if image_path:
                args["images"] = [image_path]

            # Send the query
            if stream:
                stream_response = ollama.generate(**args)
                for chunk in stream_response:
                    print(chunk["message"]["content"], end="", flush=True)
            else:
                response = ollama.generate(**args)
                print(response["message"]["content"])
        except Exception as e:
            return f"Error: {e}"


if __name__ == "__main__":
    # Initialize the client
    llava_client = OllamaLLaVAClient()

    # Example usage
    image_path = "test_images/workgroup.jpg"  # Path to an image
    prompt_file = "prompt.txt"  # Path to the prompt file
    prompt = ""
    with open(prompt_file, "r", encoding="utf-8") as file:
        prompt = file.read().strip()

    response = ollama.generate("llava:34b", prompt, images=[image_path])
    print("Description:\n" + response["response"])

    # Query the model
    # llava_client.query_model(prompt_file, image_path=image_path, stream=True)
