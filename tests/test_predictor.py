from __future__ import annotations

import pandas as pd

from src.id3 import ID3Classifier
from src.predictor import format_prediction, predict_manual_artifacts, predict_test_row_artifacts
from src.preprocessing import PreprocessConfig, PreprocessingPipeline
from src.utils import TARGET_COL


def _fit_model_and_pipe() -> tuple[ID3Classifier, PreprocessingPipeline, pd.DataFrame]:
    df = pd.DataFrame(
        {
            TARGET_COL: [0, 0, 1, 1],
            "TLD": ["com", "com", "org", "net"],
            "URLLength": [10.0, 12.0, 80.0, 100.0],
        }
    )
    pipe = PreprocessingPipeline(config=PreprocessConfig(n_bins=3, tld_top_n=3))
    pipe.fit(df)
    X_train = pipe.transform_X(df)
    model = ID3Classifier(max_depth=3, min_samples_split=2)
    model.fit(X_train, df[TARGET_COL])
    return model, pipe, df


def test_predict_test_row_artifacts_returns_transformed_row():
    model, pipe, df = _fit_model_and_pipe()
    pred, path, transformed = predict_test_row_artifacts(model, pipe, df.iloc[0])
    assert pred in {0, 1}
    assert len(path) >= 1
    assert "URLLength" in transformed.index


def test_predict_manual_artifacts_applies_updates():
    model, pipe, _ = _fit_model_and_pipe()
    pred, path, raw_row, transformed = predict_manual_artifacts(
        model,
        pipe,
        {"URLLength": 55.0, "TLD": "com"},
    )
    assert pred in {0, 1}
    assert raw_row["URLLength"] == 55.0
    assert len(path) >= 1
    assert "TLD" in transformed.index


def test_format_prediction_known_labels():
    assert format_prediction(0) == "legitimate"
    assert format_prediction(1) == "phishing"

