# pipeline/image_utils.py
import numpy as np
import cv2
from PIL import Image
import base64


def overlay_image(
    background: np.ndarray, overlay: np.ndarray, x: int, y: int
) -> np.ndarray:
    """
    Overlays an RGBA image (overlay) on a BGR background at position (x, y).
    """
    h, w = overlay.shape[:2]
    if overlay.shape[2] < 4:
        raise ValueError("Overlay image must have an alpha channel.")
    alpha = overlay[:, :, 3] / 255.0

    y1 = max(0, y)
    y2 = min(background.shape[0], y + h)
    x1 = max(0, x)
    x2 = min(background.shape[1], x + w)
    overlay_y1 = max(0, -y)
    overlay_y2 = overlay_y1 + (y2 - y1)
    overlay_x1 = max(0, -x)
    overlay_x2 = overlay_x1 + (x2 - x1)

    for c in range(3):
        background[y1:y2, x1:x2, c] = (
            alpha[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
            * overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, c]
            + (1 - alpha[overlay_y1:overlay_y2, overlay_x1:overlay_x2])
            * background[y1:y2, x1:x2, c]
        )
    return background


def rotate_with_canvas(image: np.ndarray, angle: float, extra: int = 50) -> np.ndarray:
    """
    Rotates an RGBA image by a given angle with extra padding to avoid clipping.
    """
    h, w = image.shape[:2]
    canvas = np.zeros((h + 2 * extra, w + 2 * extra, 4), dtype=image.dtype)
    canvas[extra : extra + h, extra : extra + w] = image
    center = ((w + 2 * extra) // 2, (h + 2 * extra) // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        canvas,
        rotation_matrix,
        (w + 2 * extra, h + 2 * extra),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return rotated


def read_imagefile(file_bytes: bytes) -> np.ndarray:
    """
    Converts raw file bytes into a NumPy array (BGR image).
    """
    image = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def encode_image_to_base64(image: np.ndarray) -> str:
    """
    Encodes a BGR image (JPEG format) as a base64 string with data URI header.
    """
    success, encoded_image = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Image encoding failed")
    base64_bytes = base64.b64encode(encoded_image.tobytes())
    base64_string = base64_bytes.decode("utf-8")
    return f"data:image/jpeg;base64,{base64_string}"


def draw_faces(image: np.ndarray, faces: list) -> np.ndarray:
    """
    Draws face bounding boxes and landmarks onto the image.
    Used for debugging.
    """
    output = image.copy()
    for face in faces:
        bbox = face["bbox"]
        landmarks = face["landmarks"]
        cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        for point in landmarks.values():
            cv2.circle(output, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)
    return output
