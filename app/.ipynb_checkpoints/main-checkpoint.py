"""
FastAPI Application Entrypoint.
Initializes the application instance, registers middleware, includes routers,
and attaches observability telemetry.
"""

from fastapi import FastAPI
from app.config import settings
from app.lifespan import lifespan
from app.api.v1.router import api_v1_router


def create_application() -> FastAPI:
    """Factory function to build and configure the FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Mount version 1 API endpoints
    app.include_router(api_v1_router, prefix="/api/v1")

    # TODO: Attach CORS middleware and Prometheus instrumentation
    return app


app = create_application()
