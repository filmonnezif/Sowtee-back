"""
SOWTEE Error Logger
Structured error logging with model-specific tracking.
"""

import structlog
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal
from datetime import datetime

from ..config import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class ErrorEntry:
    """A structured error log entry."""
    timestamp: str
    level: Literal["error", "warning", "info"]
    source: str  # Service/component name
    model: str | None  # AI model if applicable
    error_type: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class ErrorLogger:
    """
    Structured error logging service.
    
    Features:
    - Model-specific error tracking
    - JSON log format for analysis
    - Rate limit detection alerts
    - Performance metrics
    """
    
    def __init__(self, log_dir: str | None = None) -> None:
        self.settings = get_settings()
        self._log_dir = Path(log_dir) if log_dir else Path("./data/logs")
        self._log_dir.mkdir(parents=True, exist_ok=True)
        
        self._error_counts: dict[str, int] = {}
        self._recent_errors: list[ErrorEntry] = []
        self._max_recent = 100
    
    def _get_timestamp(self) -> str:
        """Get ISO format timestamp."""
        return datetime.utcnow().isoformat() + "Z"
    
    def _write_to_file(self, entry: ErrorEntry) -> None:
        """Write error entry to log file."""
        try:
            log_file = self._log_dir / f"errors_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
            with open(log_file, "a") as f:
                f.write(entry.to_json() + "\n")
        except Exception as e:
            logger.warning("Failed to write error to file", error=str(e))
    
    def _track_error(self, entry: ErrorEntry) -> None:
        """Track error in memory."""
        # Count by source
        key = f"{entry.source}:{entry.error_type}"
        self._error_counts[key] = self._error_counts.get(key, 0) + 1
        
        # Keep recent errors
        self._recent_errors.append(entry)
        if len(self._recent_errors) > self._max_recent:
            self._recent_errors = self._recent_errors[-self._max_recent:]
    
    def log_error(
        self,
        source: str,
        error_type: str,
        message: str,
        model: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Log an error.
        
        Args:
            source: Component/service that generated the error
            error_type: Type/category of error
            message: Human-readable error message
            model: AI model name if applicable
            details: Additional error details
        """
        entry = ErrorEntry(
            timestamp=self._get_timestamp(),
            level="error",
            source=source,
            model=model,
            error_type=error_type,
            message=message,
            details=details or {},
        )
        
        self._track_error(entry)
        self._write_to_file(entry)
        
        logger.error(
            message,
            source=source,
            error_type=error_type,
            model=model,
            **entry.details,
        )
    
    def log_warning(
        self,
        source: str,
        error_type: str,
        message: str,
        model: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log a warning."""
        entry = ErrorEntry(
            timestamp=self._get_timestamp(),
            level="warning",
            source=source,
            model=model,
            error_type=error_type,
            message=message,
            details=details or {},
        )
        
        self._track_error(entry)
        self._write_to_file(entry)
        
        logger.warning(
            message,
            source=source,
            error_type=error_type,
            model=model,
            **entry.details,
        )
    
    def log_rate_limit(
        self,
        model: str,
        source: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log a rate limit event."""
        self.log_warning(
            source=source,
            error_type="rate_limit",
            message=f"Rate limit hit for model {model}",
            model=model,
            details={
                "action": "switching_to_fallback",
                **(details or {}),
            },
        )
    
    def log_model_error(
        self,
        model: str,
        source: str,
        error: Exception,
        prompt_preview: str | None = None,
    ) -> None:
        """Log an AI model error."""
        self.log_error(
            source=source,
            error_type="model_error",
            message=f"Model {model} failed: {str(error)}",
            model=model,
            details={
                "error_class": type(error).__name__,
                "prompt_preview": prompt_preview[:100] if prompt_preview else None,
            },
        )
    
    def get_error_counts(self) -> dict[str, int]:
        """Get error counts by source:type."""
        return self._error_counts.copy()
    
    def get_recent_errors(self, limit: int = 20) -> list[dict]:
        """Get recent errors."""
        return [e.to_dict() for e in self._recent_errors[-limit:]]
    
    def get_model_errors(self, model: str, limit: int = 20) -> list[dict]:
        """Get errors for a specific model."""
        model_errors = [e for e in self._recent_errors if e.model == model]
        return [e.to_dict() for e in model_errors[-limit:]]
    
    def clear_counts(self) -> None:
        """Clear error counts (for testing/reset)."""
        self._error_counts = {}


# Singleton instance
_error_logger: ErrorLogger | None = None


def get_error_logger() -> ErrorLogger:
    """Get the error logger singleton."""
    global _error_logger
    if _error_logger is None:
        _error_logger = ErrorLogger()
    return _error_logger
