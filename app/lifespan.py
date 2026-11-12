"""
Application Lifespan Manager.
Handles startup (loading model weights into RAM once) and graceful shutdown logic.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.core.inference import ModelManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    - Startup: Initialize and pre-warm model manager.
    - Shutdown: Clean up resources and close connections.
    """
    model_manager = ModelManager(settings.MODEL_PATH)
    model_manager.load_model()
    app.state.model_manager = model_manager
    yield
    if hasattr(app.state, "model_manager") and app.state.model_manager is not None:
        app.state.model_manager.unload_model()
