#!/usr/bin/env python3
import cv2
import json
import os
import sys


def pick_hat_anchors(input_dir, metadata_file):
    """
    Iterates over all PNG files in 'input_dir', letting the user mark three points:
      1. Left border of the inner hat part.
      2. Right border of the inner hat part.
      3. Lower brim (anchor).
    Press 's' to save the points for an image, or ESC to skip.
    All metadata is saved into the JSON file.
    """
    # Load existing metadata if available.
    if os.path.isfile(metadata_file):
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Error loading existing metadata: {e}")
            metadata = {}
    else:
        metadata = {}

    # Get list of PNG files in the input directory.
    png_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".png"))
    if not png_files:
        print(f"No PNG files found in {input_dir}.")
        return

    for png_file in png_files:
        file_path = os.path.join(input_dir, png_file)
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Could not load image: {file_path}")
            continue

        base_name = os.path.splitext(png_file)[0]
        print(f"\nProcessing: {png_file}")
        print("Please click 3 points in order:")
        print("  1. Left border of inner hat part")
        print("  2. Right border of inner hat part")
        print("  3. Lower brim (anchor)")
        print("Press 's' to save points for this image, or ESC to skip.\n")

        points = []

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                print(f"Point {len(points)} selected: ({x}, {y})")

        cv2.namedWindow("Hat Anchor Picker", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Hat Anchor Picker", on_mouse)

        while True:
            disp_img = img.copy()
            # Draw circles and labels for each clicked point.
            for i, pt in enumerate(points):
                cv2.circle(disp_img, pt, 50, (0, 255, 0, 255), -1)
                cv2.putText(
                    disp_img,
                    f"{i+1}",
                    (pt[0] + 5, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            cv2.imshow("Hat Anchor Picker", disp_img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to skip this image
                print("Skipping this image; no data saved.")
                break
            if key == ord("s"):
                if len(points) < 3:
                    print("Please select all 3 points before saving.")
                    continue
                # Save the metadata for this hat.
                metadata[base_name] = {
                    "left_border": points[0],
                    "right_border": points[1],
                    "brim": points[2],
                    # You can add default values for scaling/offset if desired:
                    "scale_mode": "width",  # or "height" etc.
                    "scale_factor": 1.0,
                    "rotation_offset": 0,
                    "offset_x": 0,
                    "offset_y": 0,
                }
                print(f"Saved anchor points for {png_file}: {metadata[base_name]}")
                break

        cv2.destroyAllWindows()

    # Write the full metadata dictionary to file once done.
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nAll images processed. Metadata saved to {metadata_file}.")


def main():
    """
    Usage:
      python hat_anchor_picker.py <input_dir> [metadata_file]

    - <input_dir>: Directory containing hat PNG files.
    - [metadata_file]: (Optional) Path for the JSON metadata file;
         defaults to 'hats_metadata.json' inside the input_dir.
    """
    if len(sys.argv) < 2:
        print("Usage: python hat_anchor_picker.py <input_dir> [metadata_file]")
        sys.exit(1)

    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print(f"Invalid directory: {input_dir}")
        sys.exit(1)

    if len(sys.argv) >= 3:
        metadata_file = sys.argv[2]
    else:
        metadata_file = os.path.join(input_dir, "hats_metadata.json")

    pick_hat_anchors(input_dir, metadata_file)


if __name__ == "__main__":
    main()
