from __future__ import annotations

from io import BytesIO
from typing import BinaryIO, Literal, TypedDict

import pandas as pd

from src.data_loader import load_csv_from_path, load_csv_from_upload, resolve_default_path


DataSourceMode = Literal["default", "upload", "none", "error"]


class DataLoadResult(TypedDict):
    dataframe: pd.DataFrame | None
    source: DataSourceMode
    error: str | None


def load_dataframe(uploaded: BinaryIO | BytesIO | None) -> DataLoadResult:
    path = resolve_default_path()
    try:
        if uploaded is not None:
            return {"dataframe": load_csv_from_upload(uploaded), "source": "upload", "error": None}
        if path.is_file():
            return {"dataframe": load_csv_from_path(path), "source": "default", "error": None}
        return {"dataframe": None, "source": "none", "error": None}
    except (FileNotFoundError, ValueError) as exc:
        return {"dataframe": None, "source": "error", "error": str(exc)}
