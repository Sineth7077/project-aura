"""FastAPI application entry point for Project Aura.

Configures CORS, initializes services at startup (lifespan), registers routes,
and handles global exception logging.

Run with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

This application has NOT been deployed to any external hosting service.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.config import load_api_config
from src.api.routes import health as health_routes
from src.api.routes import inference as inference_routes
from src.api.routes import gesture as gesture_routes
from src.api.routes import agent as agent_routes

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize services on startup, clean up on shutdown.

    Model services are initialized lazily but we attempt an eager load here
    so the first request is not delayed. Failures are non-fatal: the API
    degrades gracefully and reports readiness as false.
    """
    config = app.state.config
    logger.info(
        "Starting Project Aura API (env=%s, host=%s:%d)",
        config.environment,
        config.host,
        config.port,
    )

    # Attempt to load model (non-blocking, non-fatal)
    try:
        health_routes.model_service.load_model()
        inference_routes.model_service = health_routes.model_service
    except Exception as exc:
        logger.warning("Model pre-load failed (will load lazily): %s", exc)

    yield

    # Cleanup
    try:
        health_routes.model_service.unload_model()
        gesture_routes.gesture_service.release()
    except Exception:
        pass
    logger.info("Project Aura API shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = load_api_config()

    app = FastAPI(
        title="Project Aura 2.0 API",
        description="AI Vision Control Center - Computer vision, gesture recognition, and autonomous agents",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.state.config = config

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handler for request logging
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = __import__("time").time()
        response = await call_next(request)
        elapsed_ms = (__import__("time").time() - start_time) * 1000.0
        logger.info(
            "%s %s - %d (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    # Mount routes
    app.include_router(health_routes.router)
    app.include_router(inference_routes.router)
    app.include_router(gesture_routes.router)
    app.include_router(agent_routes.router)

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. See logs for details."},
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    config = load_api_config()
    uvicorn.run(
        "src.api.main:app",
        host=config.host,
        port=config.port,
        workers=config.workers,
    )
