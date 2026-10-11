"""
Health and Readiness Probes.
- /health/live: Indicates whether the web process is running.
- /health/ready: Indicates whether the model artifact is loaded into RAM and ready for inference.
"""

from fastapi import APIRouter, Request, status
from pydantic import BaseModel

health_router = APIRouter()


class HealthResponse(BaseModel):
    status: str


@health_router.get("/live", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def liveness_probe():
    """Liveness probe for container orchestrators (Kubernetes / Docker)."""
    return HealthResponse(status="alive")


@health_router.get("/ready", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def readiness_probe(request: Request):
    """Readiness probe checking if model weights are pre-warmed in RAM."""
    # TODO: Verify request.app.state.model_manager is loaded
    return HealthResponse(status="ready")
