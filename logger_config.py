"""
Centralized logging configuration.
- Logs to console (colored) AND to rotating file
- Log files stored in /logs folder
- Max 10MB per file, keeps last 5 files
- Format: timestamp | level | module | message
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime

# ── Log folder ────────────────────────────────────────────
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")
os.makedirs(LOG_DIR, exist_ok=True)

# ── Log format ────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ── Color codes for console ───────────────────────────────
COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",
}


class ColorFormatter(logging.Formatter):
    """Adds color to console log output by level"""

    def format(self, record):
        color = COLORS.get(record.levelname, COLORS["RESET"])
        reset = COLORS["RESET"]
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


def setup_logging(level: str = "INFO") -> None:
    """
    Call once at app startup.
    Sets up root logger with:
      - Console handler (colored)
      - Rotating file handler (plain text)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers (avoid duplicate logs on reload)
    root_logger.handlers.clear()

    # ── Console handler (colored) ─────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(ColorFormatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root_logger.addHandler(console_handler)

    # ── Rotating file handler (plain text) ────────────────
    # Max 10MB per file, keep last 5 files = 50MB max disk usage
    file_handler = logging.handlers.RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root_logger.addHandler(file_handler)

    # ── Suppress noisy third-party loggers ────────────────
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        f"Logging started — level={level} | file={LOG_FILE}"
    )


def get_logger(name: str) -> logging.Logger:
    """Get a named logger. Use in every module: logger = get_logger(__name__)"""
    return logging.getLogger(name)
