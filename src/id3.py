"""Custom ID3 decision tree: entropy, information gain, recursive splits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable

import numpy as np
import pandas as pd

from src.utils import LABEL_LEGITIMATE, LABEL_PHISHING, label_to_display


def _entropy_binary(labels: np.ndarray) -> float:
    """Shannon entropy for binary labels (0/1)."""
    n = len(labels)
    if n == 0:
        return 0.0
    p1 = float(np.sum(labels == LABEL_PHISHING)) / n
    if p1 <= 0.0 or p1 >= 1.0:
        return 0.0
    p0 = 1.0 - p1
    return -p0 * np.log2(p0) - p1 * np.log2(p1)


def _majority_class(labels: np.ndarray) -> int:
    if len(labels) == 0:
        return LABEL_LEGITIMATE
    n0 = int(np.sum(labels == LABEL_LEGITIMATE))
    n1 = int(np.sum(labels == LABEL_PHISHING))
    return LABEL_PHISHING if n1 >= n0 else LABEL_LEGITIMATE


@dataclass
class ID3Node:
    is_leaf: bool
    prediction: int | None = None  # leaf class 0/1
    feature: str | None = None
    children: dict[Hashable, "ID3Node"] = field(default_factory=dict)
    majority_label: int = LABEL_LEGITIMATE
    value_counts: dict[int, int] = field(default_factory=dict)


class ID3Classifier:
    """ID3-style classifier for discrete features only (string or hashable categories)."""

    def __init__(
        self,
        max_depth: int | None = 20,
        min_samples_split: int = 50,
    ) -> None:
        self.max_depth = max_depth if max_depth is not None else 10**9
        self.min_samples_split = min_samples_split
        self.root_: ID3Node | None = None
        self.classes_: tuple[int, ...] = (LABEL_LEGITIMATE, LABEL_PHISHING)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "ID3Classifier":
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        y_arr = np.asarray(y).astype(int).ravel()
        if len(X) != len(y_arr):
            raise ValueError("X and y length mismatch")
        features = list(X.columns)
        self.root_ = self._build(X, y_arr, features, depth=0)
        return self

    def _information_gain(self, parent_y: np.ndarray, col: str, X_col: pd.Series) -> float:
        h_parent = _entropy_binary(parent_y)
        if h_parent == 0.0:
            return 0.0
        values = X_col.unique()
        n = len(parent_y)
        weighted = 0.0
        for v in values:
            mask = X_col == v
            subset = parent_y[mask.values]
            w = len(subset) / n
            weighted += w * _entropy_binary(subset)
        return h_parent - weighted

    def _best_feature(self, X: pd.DataFrame, y: np.ndarray, features: list[str]) -> tuple[str | None, float]:
        best_name: str | None = None
        best_gain = -1.0
        for col in features:
            g = self._information_gain(y, col, X[col])
            if g > best_gain:
                best_gain = g
                best_name = col
        if best_gain <= 1e-12:
            return None, 0.0
        return best_name, best_gain

    def _build(self, X: pd.DataFrame, y: np.ndarray, features: list[str], depth: int) -> ID3Node:
        majority = _majority_class(y)
        counts = {
            LABEL_LEGITIMATE: int(np.sum(y == LABEL_LEGITIMATE)),
            LABEL_PHISHING: int(np.sum(y == LABEL_PHISHING)),
        }

        if len(y) == 0:
            return ID3Node(is_leaf=True, prediction=LABEL_LEGITIMATE, majority_label=majority, value_counts=counts)

        if np.all(y == y[0]):
            return ID3Node(is_leaf=True, prediction=int(y[0]), majority_label=majority, value_counts=counts)

        if depth >= self.max_depth or not features:
            return ID3Node(is_leaf=True, prediction=majority, majority_label=majority, value_counts=counts)

        if len(y) < self.min_samples_split:
            return ID3Node(is_leaf=True, prediction=majority, majority_label=majority, value_counts=counts)

        best, gain = self._best_feature(X, y, features)
        if best is None or gain <= 0:
            return ID3Node(is_leaf=True, prediction=majority, majority_label=majority, value_counts=counts)

        remaining = [f for f in features if f != best]
        node = ID3Node(is_leaf=False, feature=best, majority_label=majority, value_counts=counts)
        for v in X[best].unique():
            mask = X[best] == v
            X_sub = X.loc[mask].reset_index(drop=True)
            y_sub = y[mask.values]
            child = self._build(X_sub, y_sub, remaining, depth + 1)
            node.children[v] = child
        return node

    def predict_one(self, row: pd.Series) -> int:
        if self.root_ is None:
            raise RuntimeError("Model not fitted")
        return self._predict_node(self.root_, row)

    def _predict_node(self, node: ID3Node, row: pd.Series) -> int:
        if node.is_leaf:
            return int(node.prediction if node.prediction is not None else node.majority_label)
        feat = node.feature
        assert feat is not None
        val = row.get(feat, None)
        if val not in node.children:
            return int(node.majority_label)
        return self._predict_node(node.children[val], row)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("Model not fitted")
        return np.array([self.predict_one(X.iloc[i]) for i in range(len(X))], dtype=int)

    def explain_path(self, row: pd.Series) -> list[tuple[str, Any, str]]:
        """List of (feature, value, next_step) for UI; last step is class."""
        if self.root_ is None:
            raise RuntimeError("Model not fitted")
        path: list[tuple[str, Any, str]] = []
        self._explain_path(self.root_, row, path)
        return path

    def _explain_path(self, node: ID3Node, row: pd.Series, out: list[tuple[str, Any, str]]) -> None:
        if node.is_leaf:
            pred = int(node.prediction if node.prediction is not None else node.majority_label)
            out.append(("class", label_to_display(pred), "leaf"))
            return
        feat = node.feature
        assert feat is not None
        val = row.get(feat, None)
        if val not in node.children:
            out.append((feat, val, f"unseen -> majority {label_to_display(node.majority_label)}"))
            pred = int(node.majority_label)
            out.append(("class", label_to_display(pred), "leaf"))
            return
        out.append((feat, val, f"branch to {feat}={val}"))
        self._explain_path(node.children[val], row, out)

    def rules_to_text(self, max_rules: int = 32) -> list[str]:
        if self.root_ is None:
            return []
        rules: list[str] = []

        def walk(n: ID3Node, conds: list[str]) -> None:
            if len(rules) >= max_rules:
                return
            if n.is_leaf:
                pred = label_to_display(int(n.prediction if n.prediction is not None else n.majority_label))
                if conds:
                    rules.append("IF " + " AND ".join(conds) + f" THEN {pred}")
                else:
                    rules.append(f"THEN {pred}")
                return
            f = n.feature
            assert f is not None
            for v, ch in n.children.items():
                walk(ch, conds + [f"{f} == {repr(v)}"])

        walk(self.root_, [])
        return rules[:max_rules]
