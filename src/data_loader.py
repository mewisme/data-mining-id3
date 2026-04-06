"""Load and validate PhiUSIIL CSV (default path or Streamlit upload)."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from src.utils import DEFAULT_CSV_PATH, FEATURE_COLUMNS, TARGET_COL


def schema_required_columns() -> list[str]:
    """Columns that must exist for load/train: `label` plus every name in `FEATURE_COLUMNS`.

    `FILENAME` and other extra columns are optional and ignored by validation.
    """
    return [TARGET_COL, *FEATURE_COLUMNS]


def validate_schema(df: pd.DataFrame) -> None:
    """Require binary target `label` and the full PhiUSIIL feature set from `FEATURE_COLUMNS`.

    Does **not** require `FILENAME`; the preprocessing pipeline drops it when present.
    """
    missing = [c for c in schema_required_columns() if c not in df.columns]
    if not missing:
        return
    if TARGET_COL in missing:
        raise ValueError(f"Missing required target column: '{TARGET_COL}'")
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    raise ValueError(
        "CSV is missing expected feature columns: "
        + ", ".join(missing_features[:10])
        + (" ..." if len(missing_features) > 10 else "")
    )


def load_csv_from_path(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset file not found: {p.resolve()}")
    try:
        df = pd.read_csv(p)
    except Exception as e:
        raise ValueError(f"Failed to read CSV from {p}: {e}") from e
    validate_schema(df)
    return df


def load_csv_from_upload(file_obj: BinaryIO | BytesIO) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_obj)
    except Exception as e:
        raise ValueError(f"Failed to parse uploaded CSV: {e}") from e
    validate_schema(df)
    return df


def resolve_default_path() -> Path:
    """Default path relative to cwd; also try project root."""
    if DEFAULT_CSV_PATH.is_file():
        return DEFAULT_CSV_PATH
    root = Path(__file__).resolve().parent.parent
    alt = root / DEFAULT_CSV_PATH
    if alt.is_file():
        return alt
    return DEFAULT_CSV_PATH
