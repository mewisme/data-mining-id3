from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.id3 import ID3Classifier


def test_fit_length_mismatch_raises():
    clf = ID3Classifier()
    X = pd.DataFrame({"a": ["x", "y"]})
    y = np.array([0])
    with pytest.raises(ValueError, match="length mismatch"):
        clf.fit(X, y)


def test_predict_before_fit_raises():
    clf = ID3Classifier()
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict(pd.DataFrame({"a": ["x"]}))


def test_explain_path_before_fit_raises():
    clf = ID3Classifier()
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.explain_path(pd.Series({"a": "x"}))


def test_rules_to_text_respects_max_rules():
    X = pd.DataFrame({"a": ["x", "x", "y", "y"], "b": ["p", "q", "p", "q"]})
    y = np.array([0, 0, 1, 1])
    clf = ID3Classifier(max_depth=5, min_samples_split=2)
    clf.fit(X, y)
    rules = clf.rules_to_text(max_rules=1)
    assert len(rules) == 1

