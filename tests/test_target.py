"""Tests for normalize_target safety (binary labels, no silent defaults)."""

import pandas as pd
import pytest

from src.preprocessing import normalize_target


def test_numeric_int_binary():
    y, m = normalize_target(pd.Series([0, 1, 0, 1]))
    assert list(y) == [0, 1, 0, 1]
    assert m == {0: 0, 1: 1}


def test_numeric_float_binary():
    y, m = normalize_target(pd.Series([0.0, 1.0, 0.0]))
    assert list(y) == [0, 1, 0]
    assert m == {0: 0, 1: 1}


def test_string_labels_mixed_case():
    y, m = normalize_target(pd.Series(["Phishing", "legitimate", "YES"]))
    assert list(y) == [1, 0, 1]


def test_unknown_string_fails():
    with pytest.raises(ValueError, match="Unsupported target label"):
        normalize_target(pd.Series(["phishing", "spam"]))


def test_nan_fails():
    with pytest.raises(ValueError, match="missing value"):
        normalize_target(pd.Series([0, 1, None]))


def test_non_binary_numeric_fails():
    with pytest.raises(ValueError, match="non-binary"):
        normalize_target(pd.Series([0, 1, 2]))


def test_non_binary_float_fails():
    with pytest.raises(ValueError, match="non-binary"):
        normalize_target(pd.Series([0.0, 0.5, 1.0]))
