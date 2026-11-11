import logging
import sys


def get_logger(name: str = "code_completion_api") -> logging.Logger:
    """Configures and returns a structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def sanitize_code_for_logging(code: str, max_chars: int = 80) -> str:
    """Safely truncates code snippets to prevent logging entire sensitive source files."""
    if not code:
        return "<empty>"
    one_line = " ".join(code.strip().splitlines())
    if len(one_line) > max_chars:
        return one_line[:max_chars] + f"... [+{len(one_line) - max_chars} chars]"
    return one_line
