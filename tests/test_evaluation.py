from __future__ import annotations

import numpy as np

from src.evaluation import confusion_matrix_df, evaluate, report_string


def test_evaluate_metrics_known_values():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    metrics = evaluate(y_true, y_pred)
    assert metrics["accuracy"] == 0.75
    assert round(metrics["precision"], 3) == 0.667
    assert metrics["recall"] == 1.0


def test_confusion_matrix_df_shape_and_counts():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    cm_df = confusion_matrix_df(y_true, y_pred)
    assert cm_df.shape == (2, 2)
    assert int(cm_df.values.sum()) == 4


def test_report_string_contains_labels():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    report = report_string(y_true, y_pred)
    assert "legitimate" in report
    assert "phishing" in report

