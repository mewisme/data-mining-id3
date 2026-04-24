"""Classification metrics using scikit-learn only (no tree estimators)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils import LABEL_LEGITIMATE, LABEL_PHISHING, label_to_display


def evaluate(y_true: np.ndarray | pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true).astype(int).ravel()
    yp = np.asarray(y_pred).astype(int).ravel()
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "precision": float(precision_score(yt, yp, pos_label=LABEL_PHISHING, zero_division=0)),
        "recall": float(recall_score(yt, yp, pos_label=LABEL_PHISHING, zero_division=0)),
        "f1": float(f1_score(yt, yp, pos_label=LABEL_PHISHING, zero_division=0)),
    }


def confusion_matrix_df(y_true: np.ndarray | pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    yt = np.asarray(y_true).astype(int).ravel()
    yp = np.asarray(y_pred).astype(int).ravel()
    cm = confusion_matrix(yt, yp, labels=[LABEL_LEGITIMATE, LABEL_PHISHING])
    idx = [label_to_display(LABEL_LEGITIMATE), label_to_display(LABEL_PHISHING)]
    return pd.DataFrame(cm, index=[f"true_{x}" for x in idx], columns=[f"pred_{x}" for x in idx])


def report_string(y_true: np.ndarray | pd.Series, y_pred: np.ndarray) -> str:
    yt = np.asarray(y_true).astype(int).ravel()
    yp = np.asarray(y_pred).astype(int).ravel()
    target_names = [label_to_display(LABEL_LEGITIMATE), label_to_display(LABEL_PHISHING)]
    return classification_report(
        yt,
        yp,
        labels=[LABEL_LEGITIMATE, LABEL_PHISHING],
        target_names=target_names,
        zero_division=0,
    )


def report_df(y_true: np.ndarray | pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    yt = np.asarray(y_true).astype(int).ravel()
    yp = np.asarray(y_pred).astype(int).ravel()
    target_names = [label_to_display(LABEL_LEGITIMATE), label_to_display(LABEL_PHISHING)]
    report = classification_report(
        yt,
        yp,
        labels=[LABEL_LEGITIMATE, LABEL_PHISHING],
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose()
