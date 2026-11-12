"""
Prediction Endpoints.
Provides both single-item (/predict) and batch (/predict/batch) inference.
Uses anyio threadpool offloading to avoid blocking the asyncio event loop.
"""

import anyio
from fastapi import APIRouter, HTTPException, Request, status

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
    Accepts patient biometric features and returns prediction (0 = Malignant, 1 = Benign)
    and confidence probability.
    """
    model_manager = getattr(request.app.state, "model_manager", None)
    if model_manager is None or model_manager.session is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or ready for inference."
        )

    pred, diag, prob = await anyio.to_thread.run_sync(
        model_manager.predict, payload.features
    )

    return PredictionResponse(
        prediction=pred,
        diagnosis=diag,
        probability=prob,
        model_version=model_manager.version,
    )


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
    model_manager = getattr(request.app.state, "model_manager", None)
    if model_manager is None or model_manager.session is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or ready for inference."
        )

    results = await anyio.to_thread.run_sync(
        model_manager.predict_batch, payload.inputs
    )

    predictions = [
        PredictionResponse(
            prediction=pred,
            diagnosis=diag,
            probability=prob,
            model_version=model_manager.version,
        )
        for pred, diag, prob in results
    ]

    return BatchPredictionResponse(
        predictions=predictions,
        total_processed=len(predictions),
    )
