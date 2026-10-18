from typing import Any, Dict, Optional
from pydantic import BaseModel


class ImageQualityResponse(BaseModel):
    quality_passed: bool
    blur_score: float
    brightness_score: float
    issues: list[str]


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    risk_level: str
    status: str
    probabilities: Dict[str, float]
    scan_type: str
    image_quality: ImageQualityResponse

    recommendation: Optional[
        Dict[str, Any]
    ] = None

    gradcam_image: Optional[str] = None
    gradcam_error: Optional[str] = None