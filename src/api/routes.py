"""API route definitions for Project Aura 2.0.

Defines route handlers and URL mappings for the REST API. Designed to
be compatible with FastAPI/Flask when deployed to production cloud
infrastructure.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List


class Route:
    """Represents a single API route."""

    def __init__(
        self,
        path: str,
        methods: List[str],
        handler: Callable,
        name: str | None = None,
    ):
        self.path = path
        self.methods = methods
        self.handler = handler
        self.name = name or handler.__name__


class Router:
    """Simple router for registering and matching API routes."""

    def __init__(self, prefix: str = ""):
        self.prefix = prefix.rstrip("/")
        self._routes: List[Route] = []

    def add_route(
        self,
        path: str,
        methods: List[str],
        handler: Callable,
        name: str | None = None,
    ) -> Route:
        """Register a new route."""
        full_path = f"{self.prefix}{path}" if self.prefix else path
        route = Route(path=full_path, methods=methods, handler=handler, name=name)
        self._routes.append(route)
        return route

    def get(self, path: str, handler: Callable, name: str | None = None) -> Route:
        """Register a GET route."""
        return self.add_route(path, ["GET"], handler, name)

    def post(self, path: str, handler: Callable, name: str | None = None) -> Route:
        """Register a POST route."""
        return self.add_route(path, ["POST"], handler, name)

    def put(self, path: str, handler: Callable, name: str | None = None) -> Route:
        """Register a PUT route."""
        return self.add_route(path, ["PUT"], handler, name)

    def delete(self, path: str, handler: Callable, name: str | None = None) -> Route:
        """Register a DELETE route."""
        return self.add_route(path, ["DELETE"], handler, name)

    def get_routes(self) -> List[Route]:
        """Return all registered routes."""
        return list(self._routes)

    def find_route(self, path: str, method: str) -> Route | None:
        """Find a matching route for a given path and HTTP method."""
        for route in self._routes:
            if route.path == path and method in route.methods:
                return route
        return None


# Default application router
app_router = Router(prefix="/api/v1")


# Route handler stubs (implemented in production deployment)
def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "project-aura"}


def predict_vision(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Vision inference endpoint handler."""
    return {"prediction": payload}


def detect_gesture(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Gesture detection endpoint handler."""
    return {"gesture": payload}


def agent_status(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Agent orchestration status endpoint."""
    return {"agents": payload}


# Register default routes
app_router.get("/health", health_check, name="health_check")
app_router.post("/vision/predict", predict_vision, name="predict_vision")
app_router.post("/gesture/detect", detect_gesture, name="detect_gesture")
app_router.get("/agents/status", agent_status, name="agent_status")


def get_router() -> Router:
    """Return the configured application router."""
    return app_router