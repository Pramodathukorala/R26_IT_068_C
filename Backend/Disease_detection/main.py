import os
import io
import uuid
import traceback
import numpy as np

from PIL import Image

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from models.model_loader import (
    get_body_model,
    get_fecal_model,
    get_input_validator_model
)

from services.input_validator import validate_input_type

from utils.gradcam import generate_gradcam
from utils.image_quality import check_image_quality
from utils.recommendation import get_recommendation

from schemas.prediction_response import PredictionResponse


# =========================================================
# CLASS NAMES
# =========================================================

FECAL_CLASSES = [
    "Coccidiosis",
    "Healthy",
    "Newcastle"
]

BODY_CLASSES = [
    "Fowlpox",
    "Healthy"
]


# =========================================================
# SETTINGS
# =========================================================

# Retained after validation and independent test evaluation
UNCERTAINTY_THRESHOLD = 0.70


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

GRADCAM_DIR = os.path.join(
    PROJECT_ROOT,
    "gradcam_outputs"
)

STATIC_DIR = os.path.join(
    PROJECT_ROOT,
    "static"
)

os.makedirs(
    GRADCAM_DIR,
    exist_ok=True
)


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Poultry Disease AI API",
    version="2.8"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =========================================================
# GRAD-CAM STATIC ACCESS
# =========================================================

app.mount(
    "/gradcam",
    StaticFiles(
        directory=GRADCAM_DIR
    ),
    name="gradcam"
)


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def prepare_image(image_bytes):

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "INVALID_IMAGE_FILE",
                "message": "The uploaded file is not a valid image."
            }
        )

    image = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# =========================================================
# CREATE RESULT
# =========================================================

def create_result(
    predicted_class,
    confidence,
    probabilities
):

    confidence_percent = round(
        confidence * 100,
        2
    )

    # -----------------------------------------------------
    # LOW CONFIDENCE REJECTION
    # -----------------------------------------------------

    if confidence < UNCERTAINTY_THRESHOLD:

        raise HTTPException(
            status_code=422,
            detail={
                "error_code":
                    "LOW_CONFIDENCE_OR_UNSUPPORTED_IMAGE",

                "message":
                    "AI cannot safely classify this image. "
                    "Please upload a valid poultry image "
                    "for the selected scan type.",

                "confidence":
                    confidence_percent,

                "action":
                    "Retake or upload a clearer image."
            }
        )

    # -----------------------------------------------------
    # RISK LEVEL
    # -----------------------------------------------------

    status = "Prediction Available"

    if predicted_class == "Healthy":

        risk_level = "Low"

    elif confidence >= 0.80:

        risk_level = "High"

    else:

        risk_level = "Medium"

    return {
        "prediction": predicted_class,
        "confidence": confidence_percent,
        "risk_level": risk_level,
        "status": status,
        "probabilities": probabilities
    }


# =========================================================
# GRAD-CAM
# =========================================================

def attach_gradcam(
    result,
    model,
    image_array,
    prefix
):

    try:

        heatmap_name = (
            f"{prefix}_{uuid.uuid4()}.jpg"
        )

        heatmap_path = os.path.join(
            GRADCAM_DIR,
            heatmap_name
        )

        generate_gradcam(
            model=model,
            image_array=image_array,
            save_path=heatmap_path
        )

        result["gradcam_image"] = (
            f"/gradcam/{heatmap_name}"
        )

        result["gradcam_error"] = None

        print(
            "GRAD-CAM SAVED:",
            heatmap_path
        )

    except Exception as error:

        print(
            "GRAD-CAM ERROR:",
            error
        )

        traceback.print_exc()

        result["gradcam_image"] = None

        result["gradcam_error"] = str(
            error
        )

    return result


# =========================================================
# DISEASE-ONLY GRAD-CAM
# =========================================================

def attach_gradcam_if_disease(
    result,
    model,
    image_array,
    prefix
):

    # -----------------------------------------------------
    # HEALTHY RESULT
    # Do not generate or save Grad-CAM
    # -----------------------------------------------------

    if result["prediction"] == "Healthy":

        result["gradcam_image"] = None
        result["gradcam_error"] = None

        print(
            "GRAD-CAM SKIPPED: Healthy prediction"
        )

        return result

    # -----------------------------------------------------
    # DISEASE RESULT
    # Generate and save Grad-CAM
    # -----------------------------------------------------

    return attach_gradcam(
        result=result,
        model=model,
        image_array=image_array,
        prefix=prefix
    )


# =========================================================
# FRONTEND HOME
# =========================================================

@app.get("/")
def home():

    index_file = os.path.join(
        STATIC_DIR,
        "index.html"
    )

    if not os.path.exists(index_file):

        raise HTTPException(
            status_code=404,
            detail="Frontend index.html was not found."
        )

    return FileResponse(
        index_file
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "OK",

        "api": "Poultry Disease AI API",

        "version": "2.8",

        "models": {
            "fecal": "Loaded",
            "body": "Loaded",
            "input_validator": "Loaded"
        },

        "features": {
            "input_validation": "Enabled",
            "image_quality_check": "Enabled",
            "uncertainty_check": "Enabled",
            "recommendation_engine": "Enabled",
            "gradcam": "Disease Only"
        }
    }


# =========================================================
# FECAL PREDICTION
# =========================================================

@app.post(
    "/predict/fecal",
    response_model=PredictionResponse
)
async def predict_fecal(
    file: UploadFile = File(...)
):

    try:

        print(
            "\n========== FECAL REQUEST START =========="
        )

        print(
            "FILE:",
            file.filename
        )

        # -------------------------------------------------
        # READ IMAGE
        # -------------------------------------------------

        image_bytes = await file.read()

        image_array = prepare_image(
            image_bytes
        )

        # -------------------------------------------------
        # IMAGE QUALITY
        # -------------------------------------------------

        quality_result = check_image_quality(
            image_array
        )

        print(
            "IMAGE QUALITY:",
            quality_result
        )

        if not quality_result[
            "quality_passed"
        ]:

            raise HTTPException(
                status_code=422,
                detail={
                    "error_code":
                        "IMAGE_QUALITY_FAILED",

                    "message":
                        "Image quality is insufficient. "
                        "Please retake the image.",

                    "image_quality":
                        quality_result
                }
            )

        # -------------------------------------------------
        # INPUT TYPE VALIDATION
        # -------------------------------------------------

        validator_model = (
            get_input_validator_model()
        )

        validation_result = validate_input_type(
            model=validator_model,
            image_array=image_array,
            expected_type="FECAL"
        )

        print(
            "INPUT VALIDATION:",
            validation_result
        )

        # -------------------------------------------------
        # FECAL DISEASE MODEL
        # -------------------------------------------------

        model = get_fecal_model()

        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        predicted_index = int(
            np.argmax(predictions)
        )

        predicted_class = (
            FECAL_CLASSES[
                predicted_index
            ]
        )

        confidence = float(
            predictions[
                predicted_index
            ]
        )

        probabilities = {

            FECAL_CLASSES[index]:
                round(
                    float(
                        predictions[index]
                    ) * 100,
                    2
                )

            for index in range(
                len(FECAL_CLASSES)
            )
        }

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        result = create_result(
            predicted_class,
            confidence,
            probabilities
        )

        result["image_quality"] = (
            quality_result
        )

        result["scan_type"] = "FECAL"

        result["recommendation"] = (
            get_recommendation(
                result["prediction"]
            )
        )

        # -------------------------------------------------
        # SAVE GRAD-CAM ONLY IF DISEASE IS DETECTED
        # -------------------------------------------------

        result = attach_gradcam_if_disease(
            result=result,
            model=model,
            image_array=image_array,
            prefix="fecal"
        )

        print(
            "FECAL RESULT:",
            result
        )

        print(
            "========== FECAL REQUEST END ==========\n"
        )

        return result

    except HTTPException:

        raise

    except Exception as error:

        print(
            "\n========== FECAL ERROR =========="
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# =========================================================
# BODY PREDICTION
# =========================================================

@app.post(
    "/predict/body",
    response_model=PredictionResponse
)
async def predict_body(
    file: UploadFile = File(...)
):

    try:

        print(
            "\n========== BODY REQUEST START =========="
        )

        print(
            "FILE:",
            file.filename
        )

        # -------------------------------------------------
        # READ IMAGE
        # -------------------------------------------------

        image_bytes = await file.read()

        image_array = prepare_image(
            image_bytes
        )

        # -------------------------------------------------
        # IMAGE QUALITY
        # -------------------------------------------------

        quality_result = check_image_quality(
            image_array
        )

        print(
            "IMAGE QUALITY:",
            quality_result
        )

        if not quality_result[
            "quality_passed"
        ]:

            raise HTTPException(
                status_code=422,
                detail={
                    "error_code":
                        "IMAGE_QUALITY_FAILED",

                    "message":
                        "Image quality is insufficient. "
                        "Please retake the image.",

                    "image_quality":
                        quality_result
                }
            )

        # -------------------------------------------------
        # INPUT TYPE VALIDATION
        # -------------------------------------------------

        validator_model = (
            get_input_validator_model()
        )

        validation_result = validate_input_type(
            model=validator_model,
            image_array=image_array,
            expected_type="BODY"
        )

        print(
            "INPUT VALIDATION:",
            validation_result
        )

        # -------------------------------------------------
        # BODY DISEASE MODEL
        # -------------------------------------------------

        model = get_body_model()

        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        predicted_index = int(
            np.argmax(predictions)
        )

        predicted_class = (
            BODY_CLASSES[
                predicted_index
            ]
        )

        confidence = float(
            predictions[
                predicted_index
            ]
        )

        probabilities = {

            BODY_CLASSES[index]:
                round(
                    float(
                        predictions[index]
                    ) * 100,
                    2
                )

            for index in range(
                len(BODY_CLASSES)
            )
        }

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        result = create_result(
            predicted_class,
            confidence,
            probabilities
        )

        result["image_quality"] = (
            quality_result
        )

        result["scan_type"] = "BODY"

        result["recommendation"] = (
            get_recommendation(
                result["prediction"]
            )
        )

        # -------------------------------------------------
        # SAVE GRAD-CAM ONLY IF DISEASE IS DETECTED
        # -------------------------------------------------

        result = attach_gradcam_if_disease(
            result=result,
            model=model,
            image_array=image_array,
            prefix="body"
        )

        print(
            "BODY RESULT:",
            result
        )

        print(
            "========== BODY REQUEST END ==========\n"
        )

        return result

    except HTTPException:

        raise

    except Exception as error:

        print(
            "\n========== BODY ERROR =========="
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )