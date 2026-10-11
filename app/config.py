"""
Configuration Management using Pydantic Settings.
Reads environment variables with sensible defaults for local development and production.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings schema."""
    APP_NAME: str = "ML Cancer Prediction API"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    
    # Model configuration
    MODEL_PATH: str = "models/cancer_model.onnx"
    FALLBACK_MODEL_PATH: str = "models/cancer_model.joblib"
    
    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
