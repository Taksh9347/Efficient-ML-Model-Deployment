"""
Observability and Telemetry Configuration.
Configures Prometheus metrics exporter and structured application logging.
"""

import logging

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator


def setup_telemetry(app: FastAPI) -> None:
    """
    Instruments FastAPI app with Prometheus metrics exporter on /metrics.
    Tracks latency histograms, request volume, and status codes.
    """
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")


def get_logger(name: str = "ml-deployment") -> logging.Logger:
    """Returns a configured structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '{"time": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
