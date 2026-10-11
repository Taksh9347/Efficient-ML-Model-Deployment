"""
API Version 1 Central Router.
Aggregates health, prediction, and metadata routes.
"""

from fastapi import APIRouter
from app.api.v1.health import health_router
from app.api.v1.predict import predict_router

api_v1_router = APIRouter()

# Register sub-routers
api_v1_router.include_router(health_router, prefix="/health", tags=["Health"])
api_v1_router.include_router(predict_router, prefix="/predict", tags=["Predictions"])
