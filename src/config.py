"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """App-level runtime toggles."""

    show_debug_errors: bool = _read_bool("PHISHING_DEBUG_ERRORS", False)
    default_row_limit: int = _read_int("PHISHING_DEFAULT_ROW_LIMIT", 8000)


settings = Settings()

