"""Centralized structured JSON logging configuration."""

import json
import logging
import sys
import traceback
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """JSON log formatter with structured fields."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "filename": record.filename,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        # Add custom extra fields (trace_id, etc.)
        for key in ("trace_id", "generation_details_id", "model_recommendation_id", "user_id"):
            value = getattr(record, key, None)
            if value is not None:
                log_entry[key] = str(value)

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "stacktrace": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_entry, default=str)


def setup_logging(level: str = "INFO") -> None:
    """Configure application-wide structured logging."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]

    # Reduce noise from third-party libraries
    for lib in ("boto3", "botocore", "s3transfer", "urllib3", "sqlalchemy.engine", "httpx"):
        logging.getLogger(lib).setLevel(logging.WARNING)


logger = logging.getLogger("neural_hub")
