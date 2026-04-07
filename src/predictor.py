"""Prediction helpers: test-row selection and manual feature input."""

from __future__ import annotations

from typing import Any, TypeAlias

import pandas as pd

from src.id3 import ID3Classifier
from src.preprocessing import PreprocessingPipeline
from src.utils import label_to_display

PathStep: TypeAlias = tuple[str, Any, str]


def predict_test_row(
    model: ID3Classifier,
    preprocessor: PreprocessingPipeline,
    raw_row: pd.Series,
) -> tuple[int, list[PathStep]]:
    """Transform one raw feature row and predict with path explanation."""
    df = raw_row.to_frame().T
    Xd = preprocessor.transform_X(df)
    pred = model.predict_one(Xd.iloc[0])
    path = model.explain_path(Xd.iloc[0])
    return int(pred), path


def predict_manual(
    model: ID3Classifier,
    preprocessor: PreprocessingPipeline,
    updates: dict[str, str | float | int],
) -> tuple[int, list[PathStep]]:
    """Fill missing features from training defaults; user updates override."""
    frame = preprocessor.manual_input_frame(updates)
    Xd = preprocessor.transform_X(frame)
    pred = model.predict_one(Xd.iloc[0])
    path = model.explain_path(Xd.iloc[0])
    return int(pred), path


def predict_test_row_artifacts(
    model: ID3Classifier,
    preprocessor: PreprocessingPipeline,
    raw_row: pd.Series,
) -> tuple[int, list[PathStep], pd.Series]:
    """Predict one test row and return transformed row used by ID3."""
    df = raw_row.to_frame().T
    Xd = preprocessor.transform_X(df)
    pred = model.predict_one(Xd.iloc[0])
    path = model.explain_path(Xd.iloc[0])
    return int(pred), path, Xd.iloc[0]


def predict_manual_artifacts(
    model: ID3Classifier,
    preprocessor: PreprocessingPipeline,
    updates: dict[str, str | float | int],
) -> tuple[int, list[PathStep], pd.Series, pd.Series]:
    """Predict manual input and return raw+transformed rows."""
    frame = preprocessor.manual_input_frame(updates)
    Xd = preprocessor.transform_X(frame)
    pred = model.predict_one(Xd.iloc[0])
    path = model.explain_path(Xd.iloc[0])
    return int(pred), path, frame.iloc[0], Xd.iloc[0]


def format_prediction(pred: int) -> str:
    return label_to_display(pred)
