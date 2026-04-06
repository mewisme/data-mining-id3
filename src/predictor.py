"""Prediction helpers: test-row selection and manual feature input."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.id3 import ID3Classifier
from src.preprocessing import PreprocessingPipeline
from src.utils import label_to_display


def predict_test_row(
    model: ID3Classifier,
    preprocessor: PreprocessingPipeline,
    raw_row: pd.Series,
) -> tuple[int, list[tuple[str, Any, str]]]:
    """Transform one raw feature row and predict with path explanation."""
    df = raw_row.to_frame().T
    Xd = preprocessor.transform_X(df)
    pred = model.predict_one(Xd.iloc[0])
    path = model.explain_path(Xd.iloc[0])
    return int(pred), path


def predict_manual(
    model: ID3Classifier,
    preprocessor: PreprocessingPipeline,
    updates: dict[str, Any],
) -> tuple[int, list[tuple[str, Any, str]]]:
    """Fill missing features from training defaults; user updates override."""
    frame = preprocessor.manual_input_frame(updates)
    Xd = preprocessor.transform_X(frame)
    pred = model.predict_one(Xd.iloc[0])
    path = model.explain_path(Xd.iloc[0])
    return int(pred), path


def format_prediction(pred: int) -> str:
    return label_to_display(pred)
