"""Custom exception hierarchy for Project Aura 2.0.

All project-specific exceptions inherit from AuraException to allow
consistent error handling across notebooks and production code.
"""


class AuraException(Exception):
    """Base exception for all Project Aura errors."""

    def __init__(self, message: str, error_code: str | None = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "AURA_UNKNOWN"

    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(AuraException):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str):
        super().__init__(message, error_code="AURA_CONFIG_ERROR")


class ModelNotFoundError(AuraException):
    """Raised when a requested model is not available."""

    def __init__(self, model_name: str):
        super().__init__(f"Model '{model_name}' not found", error_code="AURA_MODEL_NOT_FOUND")


class InferenceError(AuraException):
    """Raised when model inference fails."""

    def __init__(self, message: str):
        super().__init__(message, error_code="AURA_INFERENCE_ERROR")


class DataIngestionError(AuraException):
    """Raised when data ingestion or preprocessing fails."""

    def __init__(self, message: str):
        super().__init__(message, error_code="AURA_DATA_ERROR")


class CloudConnectionError(AuraException):
    """Raised when cloud storage connection fails."""

    def __init__(self, message: str):
        super().__init__(message, error_code="AURA_CLOUD_ERROR")


class GestureRecognitionError(AuraException):
    """Raised when gesture recognition fails."""

    def __init__(self, message: str):
        super().__init__(message, error_code="AURA_GESTURE_ERROR")


class AgentOrchestrationError(AuraException):
    """Raised when AI agent orchestration fails."""

    def __init__(self, message: str):
        super().__init__(message, error_code="AURA_AGENT_ERROR")