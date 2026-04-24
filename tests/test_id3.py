"""Custom ID3: split choice, prediction, unseen-value fallback."""

import numpy as np
import pandas as pd

from src.id3 import ID3Classifier, _entropy_binary, _majority_class
from src.utils import LABEL_LEGITIMATE, LABEL_PHISHING


def test_entropy_binary_pure_and_mixed():
    assert _entropy_binary(np.array([0, 0, 0])) == 0.0
    assert _entropy_binary(np.array([1, 1])) == 0.0
    h = _entropy_binary(np.array([0, 1]))
    assert abs(h - 1.0) < 1e-9


def test_majority_class_tie_prefers_phishing():
    y = np.array([0, 0, 1, 1])
    assert _majority_class(y) == LABEL_PHISHING


def test_id3_splits_on_higher_information_gain_feature():
    X = pd.DataFrame({"a": ["x", "x", "y", "y"], "b": ["p", "q", "p", "q"]})
    y = np.array([0, 0, 1, 1])
    clf = ID3Classifier(max_depth=5, min_samples_split=2)
    clf.fit(X, pd.Series(y))
    assert clf.root_ is not None and clf.root_.feature == "a"


def test_predict_follows_leaves():
    X = pd.DataFrame({"a": ["x", "x", "y", "y"]})
    y = np.array([0, 0, 1, 1])
    clf = ID3Classifier(max_depth=5, min_samples_split=2)
    clf.fit(X, pd.Series(y))
    assert clf.predict_one(pd.Series({"a": "x"})) == LABEL_PHISHING
    assert clf.predict_one(pd.Series({"a": "y"})) == LABEL_LEGITIMATE


def test_unseen_categorical_uses_majority_at_node():
    X = pd.DataFrame({"a": ["x", "x", "y", "y"]})
    y = np.array([0, 0, 1, 1])
    clf = ID3Classifier(max_depth=5, min_samples_split=2)
    clf.fit(X, pd.Series(y))
    pred = clf.predict_one(pd.Series({"a": "never_seen"}))
    assert pred == _majority_class(y)
