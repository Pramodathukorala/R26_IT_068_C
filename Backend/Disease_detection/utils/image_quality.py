import cv2
import numpy as np


def check_image_quality(image_array):

    # Remove batch dimension
    image = image_array[0]

    # Ensure uint8 format
    image = np.clip(
        image,
        0,
        255
    ).astype(np.uint8)

    # Convert RGB to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # ==========================================
    # BLUR DETECTION
    # ==========================================

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    # ==========================================
    # BRIGHTNESS DETECTION
    # ==========================================

    brightness_score = float(
        np.mean(gray)
    )

    # ==========================================
    # QUALITY DECISION
    # ==========================================

    issues = []

    # Temporary development thresholds.
    # These will later be calibrated using
    # real validation images.

    if blur_score < 80:
        issues.append(
            "Image is too blurry"
        )

    if brightness_score < 40:
        issues.append(
            "Image is too dark"
        )

    if brightness_score > 215:
        issues.append(
            "Image is too bright"
        )

    quality_passed = (
        len(issues) == 0
    )

    return {
        "quality_passed": quality_passed,
        "blur_score": round(
            float(blur_score),
            2
        ),
        "brightness_score": round(
            brightness_score,
            2
        ),
        "issues": issues
    }