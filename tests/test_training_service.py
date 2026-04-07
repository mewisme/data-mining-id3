from __future__ import annotations

import pandas as pd
import pytest

from src.services.training_service import run_training
from src.utils import TARGET_COL


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            TARGET_COL: [0, 0, 1, 1, 0, 1],
            "TLD": ["com", "com", "org", "net", "com", "org"],
            "URLLength": [10.0, 11.0, 50.0, 100.0, 12.0, 75.0],
        }
    )


def test_run_training_returns_expected_artifacts():
    artifacts = run_training(
        _toy_df(),
        n_bins=4,
        bin_strategy="quantile",
        tld_top_n=5,
        test_size=0.33,
        max_depth=4,
        min_samples_split=2,
        row_limit=0,
    )
    expected = {"pipe", "model", "train_df", "test_df", "X_train", "X_test", "y_test", "split_info", "sampling_info"}
    assert expected.issubset(set(artifacts.keys()))
    assert artifacts["split_info"]["rows_total"] == len(_toy_df())
    assert artifacts["split_info"]["rows_train"] + artifacts["split_info"]["rows_test"] == len(_toy_df())


def test_run_training_row_limit_applies():
    artifacts = run_training(
        _toy_df(),
        n_bins=3,
        bin_strategy="uniform",
        tld_top_n=5,
        test_size=0.5,
        max_depth=3,
        min_samples_split=2,
        row_limit=4,
    )
    assert artifacts["sampling_info"]["row_sampling_enabled"] is True
    assert artifacts["sampling_info"]["rows_used_for_training_pipeline"] == 4


def test_run_training_fails_for_too_few_rows():
    df = pd.DataFrame({TARGET_COL: [0], "TLD": ["com"], "URLLength": [10.0]})
    with pytest.raises(ValueError, match="at least 2 rows"):
        run_training(
            df,
            n_bins=3,
            bin_strategy="quantile",
            tld_top_n=5,
            test_size=0.2,
            max_depth=3,
            min_samples_split=2,
            row_limit=0,
        )


def test_run_training_falls_back_without_stratify_failure():
    # One class has only one sample -> cannot stratify with common test sizes
    df = pd.DataFrame(
        {
            TARGET_COL: [0, 0, 0, 1],
            "TLD": ["com", "com", "org", "net"],
            "URLLength": [10.0, 12.0, 14.0, 90.0],
        }
    )
    artifacts = run_training(
        df,
        n_bins=3,
        bin_strategy="quantile",
        tld_top_n=3,
        test_size=0.5,
        max_depth=3,
        min_samples_split=2,
        row_limit=0,
    )
    assert artifacts["split_info"]["rows_total"] == 4

