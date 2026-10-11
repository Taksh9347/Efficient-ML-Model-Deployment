"""
Pydantic Request and Response Schemas.
Enforces strict type validation and boundary checks for the Breast Cancer dataset.
"""

from pydantic import BaseModel, Field
from typing import List


class CancerFeatures(BaseModel):
    """
    Biological biometric feature schema corresponding to the Breast Cancer Wisconsin dataset.
    """
    mean_radius: float = Field(..., gt=0, example=17.99, description="Mean radius of tumor cells")
    mean_texture: float = Field(..., gt=0, example=10.38, description="Standard deviation of gray-scale values")
    mean_perimeter: float = Field(..., gt=0, example=122.8, description="Mean perimeter")
    mean_area: float = Field(..., gt=0, example=1001.0, description="Mean area")
    mean_smoothness: float = Field(..., gt=0, le=1.0, example=0.1184, description="Local variation in radius lengths")
    # TODO: Add remaining 25 features or relevant subset as needed by the model


class PredictionRequest(BaseModel):
    """Single prediction request payload."""
    features: CancerFeatures


class PredictionResponse(BaseModel):
    """Single prediction response payload."""
    prediction: int = Field(..., description="0 = Benign, 1 = Malignant")
    diagnosis: str = Field(..., example="Malignant")
    probability: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    model_version: str


class BatchPredictionRequest(BaseModel):
    """Batch prediction request payload."""
    inputs: List[CancerFeatures] = Field(..., min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    """Batch prediction response payload."""
    predictions: List[PredictionResponse]
    total_processed: int
