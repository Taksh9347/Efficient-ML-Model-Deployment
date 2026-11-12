"""
Pydantic Request and Response Schemas.
Enforces strict type validation and boundary checks for the Breast Cancer dataset.
"""


from pydantic import BaseModel, Field


class CancerFeatures(BaseModel):
    """
    Biological biometric feature schema corresponding to the Breast Cancer Wisconsin dataset (30 features).
    The top 5 mean features are required; the remaining 25 have default median values
    so standard payloads with 5 features remain 100% valid.
    """
    mean_radius: float = Field(..., gt=0, json_schema_extra={"example": 17.99}, description="Mean radius of tumor cells")
    mean_texture: float = Field(..., gt=0, json_schema_extra={"example": 10.38}, description="Standard deviation of gray-scale values")
    mean_perimeter: float = Field(..., gt=0, json_schema_extra={"example": 122.8}, description="Mean perimeter")
    mean_area: float = Field(..., gt=0, json_schema_extra={"example": 1001.0}, description="Mean area")
    mean_smoothness: float = Field(..., gt=0, le=1.0, json_schema_extra={"example": 0.1184}, description="Local variation in radius lengths")
    mean_compactness: float = Field(default=0.0926, gt=0, description="Mean compactness")
    mean_concavity: float = Field(default=0.0615, ge=0, description="Mean concavity")
    mean_concave_points: float = Field(default=0.0335, ge=0, description="Mean concave points")
    mean_symmetry: float = Field(default=0.1792, gt=0, description="Mean symmetry")
    mean_fractal_dimension: float = Field(default=0.0615, gt=0, description="Mean fractal dimension")
    radius_error: float = Field(default=0.3242, gt=0, description="Radius error")
    texture_error: float = Field(default=1.108, gt=0, description="Texture error")
    perimeter_error: float = Field(default=2.287, gt=0, description="Perimeter error")
    area_error: float = Field(default=24.53, gt=0, description="Area error")
    smoothness_error: float = Field(default=0.0064, ge=0, description="Smoothness error")
    compactness_error: float = Field(default=0.0204, ge=0, description="Compactness error")
    concavity_error: float = Field(default=0.0259, ge=0, description="Concavity error")
    concave_points_error: float = Field(default=0.0109, ge=0, description="Concave points error")
    symmetry_error: float = Field(default=0.0187, ge=0, description="Symmetry error")
    fractal_dimension_error: float = Field(default=0.0032, ge=0, description="Fractal dimension error")
    worst_radius: float = Field(default=14.97, gt=0, description="Worst radius")
    worst_texture: float = Field(default=25.41, gt=0, description="Worst texture")
    worst_perimeter: float = Field(default=97.66, gt=0, description="Worst perimeter")
    worst_area: float = Field(default=686.5, gt=0, description="Worst area")
    worst_smoothness: float = Field(default=0.1313, gt=0, description="Worst smoothness")
    worst_compactness: float = Field(default=0.2119, ge=0, description="Worst compactness")
    worst_concavity: float = Field(default=0.2267, ge=0, description="Worst concavity")
    worst_concave_points: float = Field(default=0.0999, ge=0, description="Worst concave points")
    worst_symmetry: float = Field(default=0.2822, gt=0, description="Worst symmetry")
    worst_fractal_dimension: float = Field(default=0.08, gt=0, description="Worst fractal dimension")

    def to_features_list(self) -> list[float]:
        """Returns the 30 features ordered exactly as expected by the trained model."""
        return [
            self.mean_radius,
            self.mean_texture,
            self.mean_perimeter,
            self.mean_area,
            self.mean_smoothness,
            self.mean_compactness,
            self.mean_concavity,
            self.mean_concave_points,
            self.mean_symmetry,
            self.mean_fractal_dimension,
            self.radius_error,
            self.texture_error,
            self.perimeter_error,
            self.area_error,
            self.smoothness_error,
            self.compactness_error,
            self.concavity_error,
            self.concave_points_error,
            self.symmetry_error,
            self.fractal_dimension_error,
            self.worst_radius,
            self.worst_texture,
            self.worst_perimeter,
            self.worst_area,
            self.worst_smoothness,
            self.worst_compactness,
            self.worst_concavity,
            self.worst_concave_points,
            self.worst_symmetry,
            self.worst_fractal_dimension,
        ]


class PredictionRequest(BaseModel):
    """Single prediction request payload."""
    features: CancerFeatures


class PredictionResponse(BaseModel):
    """Single prediction response payload."""
    prediction: int = Field(..., description="0 = Malignant, 1 = Benign")
    diagnosis: str = Field(..., json_schema_extra={"example": "Malignant"})
    probability: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    model_version: str


class BatchPredictionRequest(BaseModel):
    """Batch prediction request payload."""
    inputs: list[CancerFeatures] = Field(..., min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    """Batch prediction response payload."""
    predictions: list[PredictionResponse]
    total_processed: int
