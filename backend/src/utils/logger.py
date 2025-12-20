import logging
from typing import Any
import json
from datetime import datetime


class CustomFormatter(logging.Formatter):
    """Custom formatter to add JSON structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "levelname", "levelno", "pathname",
                          "filename", "module", "lineno", "funcName", "created",
                          "msecs", "relativeCreated", "thread", "threadName",
                          "processName", "process", "getMessage", "exc_info",
                          "exc_text", "stack_info"]:
                log_entry[key] = value

        return json.dumps(log_entry)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with custom formatting"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter())

    logger.addHandler(handler)
    logger.propagate = False  # Prevent duplicate logs

    return logger


# Global logger instance
app_logger = setup_logger("rag_chatbot")