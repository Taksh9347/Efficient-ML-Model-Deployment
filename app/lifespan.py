"""
Application Lifespan Manager.
Handles startup (loading model weights into RAM once) and graceful shutdown logic.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    - Startup: Initialize and pre-warm model manager.
    - Shutdown: Clean up resources and close connections.
    """
    # TODO: Initialize ModelManager and attach to app.state.model_manager
    # app.state.model_manager = ModelManager(settings.MODEL_PATH)
    # app.state.model_manager.load_model()
    yield
    # TODO: Clean up model resources on shutdown if necessary
