import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the asset directories
ASSET_DIRS = {
    "backgrounds": "assets/backgrounds",
    "hats": "assets/hats",
    "glasses": "assets/glasses",
    "effects": "assets/effects",
    "masks": "assets/masks",
}
# We'll store our ImagePipeline instance here.
app_pipeline = None


@asynccontextmanager
async def load_models_and_pipeline(app: FastAPI):
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    from pipeline.face_detection import FaceDetector
    from pipeline.background_removal import BackgroundRemover
    from pipeline.accessory_application import AccessoryPlacer
    from pipeline.openai_client import OpenAI_Client
    from pipeline import global_vars
    from pipeline.image_pipeline import ImagePipeline

    # Load global models and assign them in the global_vars module.
    global_vars.face_detector = FaceDetector(device=device)
    global_vars.bg_remover = BackgroundRemover(
        model_path="checkpoints/modnet_photographic_portrait_matting.ckpt",
        device=device,
    )
    global_vars.accessory_placer = AccessoryPlacer(ASSET_DIRS)
    openai_api_key = os.getenv("OPEN_AI_API_KEY", "")
    global_vars.openai_client = OpenAI_Client(openai_api_key, model="gpt-4o")

    # Now instantiate the ImagePipeline with the required parameters.
    global app_pipeline
    app_pipeline = ImagePipeline(asset_dirs=ASSET_DIRS, device=device)
    print("Models and pipeline loaded successfully.")
    yield


app = FastAPI(lifespan=load_models_and_pipeline)


@app.post("/process_image")
async def process_image_endpoint(
    image: UploadFile = File(...),
    background_override: str = Form(None),
    effect_override: str = Form(None),
):
    """
    Receives an image and optional background/effect overrides, processes it
    through the ImagePipeline, and returns the processed images as base64 strings.
    """
    try:
        file_bytes = await image.read()
        from pipeline.image_utils import read_imagefile, encode_image_to_base64

        input_image = read_imagefile(file_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=("Error reading the uploaded image. " + str(e))
        )

    try:
        # Use the ImagePipeline instance that was created at startup.
        if app_pipeline is not None:
            results = app_pipeline.process_image(
                input_image, background_override, effect_override
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Pipeline processing error: {str(e)}"
        )

    encoded_results = [encode_image_to_base64(img) for img in results]
    return JSONResponse(content={"results": encoded_results})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
