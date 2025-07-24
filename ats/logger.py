import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

from dotenv import load_dotenv


load_dotenv()

# Configure root logger, otherwise gensim logger doesn't work
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_logger(
    name: str = "ats",
    level: Optional[int] = None,
    log_format: Optional[str] = None,
    log_file: str = "app.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB per log file
    backup_count: int = 3,  # keep 3 backup files
) -> logging.Logger:
    """
    Configure and return a logger with specified settings.

    Args:
        name: Logger name
        level: Logging level
        log_format: Custom log format string
        log_file: Path to log file (default: pipeline.log)
        max_bytes: Maximum size of each log file in bytes (default: 10 MB)
        backup_count: Number of backup files to keep (default: 3)

    Returns:
        Configured logger instance
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", logging.INFO)
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)

    if log_format is None:
        log_format = "[%(asctime)s] %(levelname)s [%(name)s] %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # logger.propagate = False  # allow propagation to ensure logging works

    # Check if custom handlers are already set for this logger, not inherited ones.
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Use RotatingFileHandler to limit log file sizes
        file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(log_format)
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger