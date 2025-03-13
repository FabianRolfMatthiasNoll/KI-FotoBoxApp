import os
import glob
from PIL import Image


def crop_transparent(image):
    """
    Crops out the transparent border from an image.
    If the image doesn't have an alpha channel, it is converted to RGBA.
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    # Extract the alpha channel
    alpha = image.split()[-1]
    # Get the bounding box of non-transparent pixels
    bbox = alpha.getbbox()
    if bbox:
        return image.crop(bbox)
    # If no non-transparent area is found, return the image as is
    return image


def process_images(input_folder, output_folder):
    """
    Processes all PNG images in input_folder, crops them,
    and saves the results in output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    png_files = glob.glob(os.path.join(input_folder, "*.png"))
    if not png_files:
        print("No PNG images found in the folder:", input_folder)
        return
    for file_path in png_files:
        try:
            with Image.open(file_path) as im:
                cropped_im = crop_transparent(im)
                base_name = os.path.basename(file_path)
                output_path = os.path.join(output_folder, base_name)
                cropped_im.save(output_path)
                print(f"Processed: {file_path} -> {output_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    # Replace these paths with your actual folder paths
    input_folder = "./input_hats"
    output_folder = "./hats"
    process_images(input_folder, output_folder)
