from pipeline.face_detection import FaceDetector
from pipeline.background_removal import BackgroundRemover
from pipeline.accessory_application import AccessoryPlacer
from pipeline.openai_client import OpenAI_Client

face_detector: FaceDetector = None  # type: ignore
bg_remover: BackgroundRemover = None  # type: ignore
accessory_placer: AccessoryPlacer = None  # type: ignore
openai_client: OpenAI_Client = None  # type: ignore
