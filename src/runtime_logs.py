from __future__ import annotations

import logging
from collections import deque
from typing import Any

MAX_LOG_LINES = 500
DEFAULT_LOG_LIMIT = 50
CAPTURED_LOG_LEVEL = logging.INFO

_backend_log_buffer: deque[str] = deque(maxlen=MAX_LOG_LINES)
_configured = False


def normalize_log_limit(value: Any) -> int:
    if isinstance(value, bool):
        return DEFAULT_LOG_LIMIT
    if isinstance(value, (int, float)):
        return max(1, min(500, int(value)))
    return DEFAULT_LOG_LIMIT


class InMemoryLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        _backend_log_buffer.append(message)


def configure_runtime_log_capture() -> None:
    global _configured
    if _configured:
        return

    handler = InMemoryLogHandler(level=CAPTURED_LOG_LEVEL)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))

    logger_names = ["", "__main__", "main", "streaming", "tool_executor", "runtime_logs", "quart.app", "uvicorn", "uvicorn.error", "uvicorn.access"]
    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        if logger.level == logging.NOTSET or logger.level > CAPTURED_LOG_LEVEL:
            logger.setLevel(CAPTURED_LOG_LEVEL)
        logger.addHandler(handler)

    _configured = True


def get_backend_logs(limit: int = DEFAULT_LOG_LIMIT) -> dict[str, Any]:
    normalized_limit = normalize_log_limit(limit)
    lines = list(_backend_log_buffer)[-normalized_limit:]
    return {
        "system": "backend",
        "limit": normalized_limit,
        "lines": lines,
        "line_count": len(lines),
    }
