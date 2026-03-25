"""Custom exception classes for domain-specific errors."""


class NeuralHubBaseError(Exception):
    """Base exception for all Neural Hub errors."""

    def __init__(self, message: str, error_code: str) -> None:
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class GenerationDetailsNotFoundError(NeuralHubBaseError):
    """Raised when generation details record is not found."""

    def __init__(self, generation_details_id: str) -> None:
        super().__init__(
            message=f"Generation details not found for id: {generation_details_id}",
            error_code="GENERATION_NOT_FOUND",
        )


class ModelRecommendationNotFoundError(NeuralHubBaseError):
    """Raised when model recommendation record is not found."""

    def __init__(self, model_recommendation_id: str) -> None:
        super().__init__(
            message=f"Model recommendation not found for id: {model_recommendation_id}",
            error_code="MODEL_NOT_FOUND",
        )


class S3OperationError(NeuralHubBaseError):
    """Raised when an S3 operation fails."""

    def __init__(self, message: str, error_code: str = "S3_DOWNLOAD_FAILED") -> None:
        super().__init__(message=message, error_code=error_code)


class DatabaseError(NeuralHubBaseError):
    """Raised when a database operation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, error_code="DATABASE_ERROR")


class KiroIntegrationError(NeuralHubBaseError):
    """Raised when Kiro integration fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, error_code="KIRO_INTEGRATION_FAILED")
