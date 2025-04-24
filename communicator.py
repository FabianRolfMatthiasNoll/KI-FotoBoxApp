import requests
import base64
import cv2
import numpy as np
import matplotlib.pyplot as plt


def test_upload(
    image_path, background_override=None, effect_override=None, accessory_override=None
):
    url = "http://localhost:8000/process_image"

    # Open the image file in binary mode
    with open(image_path, "rb") as img_file:
        files = {"image": img_file}
        # Optional form data for background and effect override:
        data = {}
        if background_override:
            data["background_override"] = background_override
        if effect_override:
            data["effect_override"] = effect_override
        if accessory_override:
            data["accessory_override"] = accessory_override

        print("Sending request to the server...")
        response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            # Get the list of base64 strings from the response
            images_base64 = result.get("results", [])
            images = []

            for img_str in images_base64:
                # Remove data URL header if present
                if img_str.startswith("data:image"):
                    header, encoded = img_str.split(",", 1)
                else:
                    encoded = img_str
                # Decode base64 to bytes
                img_data = base64.b64decode(encoded)
                # Convert bytes to a NumPy array
                nparr = np.frombuffer(img_data, np.uint8)
                # Decode the image (OpenCV reads in BGR format)
                img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img_np is None:
                    continue
                # Convert BGR to RGB (for correct display in matplotlib)
                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                images.append(img_rgb)

            # Display the resulting images using matplotlib
            num_images = len(images)
            if num_images == 0:
                print("No images to display.")
                return

            fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
            if num_images == 1:
                axes = [axes]
            for idx, ax in enumerate(axes):
                ax.imshow(images[idx])
                ax.set_title(f"Result {idx + 1}")
                ax.axis("off")

            plt.tight_layout()
            plt.show()
        else:
            print(f"Error: {response.status_code}")
            print(response.text)


if __name__ == "__main__":
    # Specify the path to your test image.
    image_path = "assets/test_images/four-friends.jpg"

    # Optionally, specify overrides (or leave as None to use default suggestions).
    accessory_override = "none"  # "none" disables, None enables
    background_override = "none"  # "none" disables, None enables, "dhbw" uses dhbw background
    effect_override = "none"  # "none" disables, None enables, "dhbw_banner" uses dhbw banner

    test_upload(image_path, background_override, effect_override, accessory_override)
