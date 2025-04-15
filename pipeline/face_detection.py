# pipeline/face_detection.py
import numpy as np
from retinaface import RetinaFace


class FaceDetector:
    """
    Uses RetinaFace to detect faces and extract landmarks.
    """

    def __init__(self, device=None):
        self.device = device

    def detect_faces(self, image: np.ndarray) -> list:
        """
        Returns a list of dictionaries each containing 'bbox' and 'landmarks' for a detected face.
        """
        faces_dict = RetinaFace.detect_faces(image)
        faces = []
        for face_id, data in faces_dict.items():
            bbox = data.get("facial_area", [])
            landmarks = data.get("landmarks", {})
            faces.append({"bbox": bbox, "landmarks": landmarks})
        return faces
