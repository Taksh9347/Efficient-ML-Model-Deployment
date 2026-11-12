"""
FastAPI Application Entrypoint.
Initializes the application instance, registers middleware, includes routers,
and attaches observability telemetry.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api.v1.predict import predict_single
from app.api.v1.router import api_v1_router
from app.config import settings
from app.core.schemas import PredictionRequest, PredictionResponse
from app.core.telemetry import setup_telemetry
from app.lifespan import lifespan


def create_application() -> FastAPI:
    """Factory function to build and configure the FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Enable CORS for cross-origin frontend requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Redirect root GET to interactive Swagger UI documentation
    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/docs")

    # OpenFaaS Root POST invocation alias for direct serverless function calls
    @app.post("/", response_model=PredictionResponse, include_in_schema=False)
    async def root_predict(payload: PredictionRequest, request: Request):
        return await predict_single(payload, request)

    # OpenFaaS & Kubernetes standard health & readiness probes
    @app.get("/_/health", include_in_schema=False)
    @app.get("/healthz", include_in_schema=False)
    async def openfaas_liveness():
        return {"status": "OK"}

    @app.get("/_/ready", include_in_schema=False)
    async def openfaas_readiness():
        return {"status": "OK"}

    # Mount version 1 API endpoints
    app.include_router(api_v1_router, prefix="/api/v1")

    # Attach Prometheus metrics exporter (/metrics)
    setup_telemetry(app)

    return app


app = create_application()
