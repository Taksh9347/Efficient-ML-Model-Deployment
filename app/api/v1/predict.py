"""
Prediction Endpoints.
Provides both single-item (/predict) and batch (/predict/batch) inference.
Uses anyio threadpool offloading to avoid blocking the asyncio event loop.
"""

from fastapi import APIRouter, Request, status

from app.core.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)

predict_router = APIRouter()


@predict_router.post(
    "",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Single cancer classification prediction"
)
async def predict_single(payload: PredictionRequest, request: Request):
    """
    Accepts patient biometric features and returns prediction (0 = Benign, 1 = Malignant)
    and confidence probability.
    """
    # TODO: Call model manager asynchronously via anyio.to_thread.run_sync
    # model_manager = request.app.state.model_manager
    # prediction, probability = await anyio.to_thread.run_sync(model_manager.predict, payload.features)


@predict_router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch cancer classification prediction"
)
async def predict_batch(payload: BatchPredictionRequest, request: Request):
    """
    Accepts a list of feature sets and returns a list of classification predictions.
    """
    # TODO: Call batch prediction asynchronously via anyio.to_thread.run_sync
