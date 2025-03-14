#!/usr/bin/env python3
import cv2
import json
import os
import sys


def pick_anchors_for_directory(input_dir, metadata_file):
    """
    Iterates over all PNGs in 'input_dir', letting the user click exactly one
    anchor point per image. Press 's' to save the anchor for that image, or ESC
    to skip. After processing all files, all metadata is saved to 'metadata_file'.
    """
    # 1) Load or create the metadata dictionary
    if os.path.isfile(metadata_file):
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"Could not parse existing metadata. Starting fresh.")
            metadata = {}
    else:
        metadata = {}

    # 2) Collect all PNG files in the directory
    png_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".png"))
    if not png_files:
        print(f"No PNG files found in {input_dir}.")
        return

    # 3) Process each PNG
    for png_file in png_files:
        file_path = os.path.join(input_dir, png_file)
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Cannot load image: {file_path}")
            continue

        base_name = os.path.splitext(png_file)[0]
        print(f"\nProcessing: {png_file}")
        print("Click once on the anchor point (e.g., glasses bridge).")
        print("Press 's' to save anchor for this image, ESC to skip.\n")

        anchor_points = []

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                anchor_points.clear()  # Ensure only one point is stored
                anchor_points.append((x, y))
                print(f"Anchor selected: (x={x}, y={y})")

        cv2.namedWindow("Anchor Picker", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Anchor Picker", on_mouse)

        while True:
            display_img = img.copy()
            if anchor_points:
                cv2.circle(display_img, anchor_points[0], 25, (0, 255, 0, 255), -1)
            cv2.imshow("Anchor Picker", display_img)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC key: skip this image
                print("Skipping this image. No anchor saved.")
                break
            elif key == ord("s"):
                if anchor_points:
                    ax, ay = anchor_points[0]
                    metadata[base_name] = {
                        "anchor_x": ax,
                        "anchor_y": ay,
                        "scale_mode": "head_width",
                        "scale_factor": 1.0,
                        "rotation_offset": 0,
                        "offset_x": 0,
                        "offset_y": 0,
                    }
                    print(f"Anchor for '{png_file}' set to: (x={ax}, y={ay})")
                else:
                    print("No anchor selected. Nothing saved for this image.")
                break

        cv2.destroyAllWindows()

    # 4) Write the entire metadata dictionary to file once processing is complete.
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nAll PNG files processed. Metadata saved to {metadata_file}.")


def main():
    """
    Usage:
        python anchor_picker.py <input_dir> [metadata_file]

    - <input_dir> : Directory containing PNG files.
    - [metadata_file]: Optional path for JSON metadata; defaults to 'glasses_metadata.json' in <input_dir>.
    """
    if len(sys.argv) < 2:
        print("Usage: python anchor_picker.py <input_dir> [metadata_file]")
        sys.exit(1)

    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print(f"Invalid directory: {input_dir}")
        sys.exit(1)

    if len(sys.argv) >= 3:
        metadata_file = sys.argv[2]
    else:
        metadata_file = os.path.join(input_dir, "glasses_metadata.json")

    pick_anchors_for_directory(input_dir, metadata_file)


if __name__ == "__main__":
    main()
