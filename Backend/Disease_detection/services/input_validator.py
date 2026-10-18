import numpy as np
from fastapi import HTTPException


# =========================================================
# INPUT VALIDATOR CLASSES
# =========================================================

VALIDATOR_CLASSES = [
    "BODY",
    "FECAL",
    "OTHER"
]


# =========================================================
# VALIDATE IMAGE TYPE
# =========================================================

def validate_input_type(
    model,
    image_array,
    expected_type
):

    # -----------------------------------------------------
    # AI PREDICTION
    # -----------------------------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_type = VALIDATOR_CLASSES[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    confidence_percent = round(
        confidence * 100,
        2
    )


    # -----------------------------------------------------
    # PROBABILITY OUTPUT
    # -----------------------------------------------------

    probabilities = {
        class_name: round(
            float(predictions[index]) * 100,
            2
        )
        for index, class_name in enumerate(
            VALIDATOR_CLASSES
        )
    }


    # -----------------------------------------------------
    # REJECT OTHER IMAGE
    # -----------------------------------------------------

    if predicted_type == "OTHER":

        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "UNSUPPORTED_IMAGE_TYPE",
                "message": (
                    "This image does not appear to be "
                    "a valid poultry body or fecal image."
                ),
                "detected_type": predicted_type,
                "validator_confidence": confidence_percent,
                "probabilities": probabilities,
                "action": (
                    "Please upload a valid poultry "
                    "body or fecal image."
                )
            }
        )


    # -----------------------------------------------------
    # REJECT WRONG SCAN TYPE
    # -----------------------------------------------------

    if predicted_type != expected_type:

        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "WRONG_IMAGE_TYPE",
                "message": (
                    f"This appears to be a "
                    f"{predicted_type.lower()} image, "
                    f"but the selected scan requires "
                    f"a {expected_type.lower()} image."
                ),
                "detected_type": predicted_type,
                "expected_type": expected_type,
                "validator_confidence": confidence_percent,
                "probabilities": probabilities,
                "action": (
                    f"Please upload a valid "
                    f"{expected_type.lower()} image."
                )
            }
        )


    # -----------------------------------------------------
    # VALID IMAGE
    # -----------------------------------------------------

    return {
        "validated": True,
        "detected_type": predicted_type,
        "confidence": confidence_percent,
        "probabilities": probabilities
    }