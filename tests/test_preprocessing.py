"""PreprocessingPipeline: train-only fit, categoricals, dropped high-cardinality columns."""

import pandas as pd

import src.preprocessing as preprocessing
from src.preprocessing import PreprocessConfig, PreprocessingPipeline, normalize_target
from src.utils import HIGH_CARD_TEXT_COLS, TARGET_COL


def _tiny_train_df():
    return pd.DataFrame(
        {
            TARGET_COL: [0, 0, 1, 1],
            "TLD": ["com", "com", "org", "net"],
            "URLLength": [10.0, 11.0, 50.0, 100.0],
        }
    )


def _tiny_test_df():
    return pd.DataFrame(
        {
            TARGET_COL: [1],
            "TLD": ["never_seen_tld"],
            "URLLength": [12.0],
        }
    )


def test_unseen_categorical_maps_to_other():
    cfg = PreprocessConfig(tld_top_n=2, n_bins=3, drop_high_card_text=True)
    pipe = PreprocessingPipeline(config=cfg)
    train = _tiny_train_df()
    test = _tiny_test_df()
    y_train, _ = normalize_target(train[TARGET_COL])
    train = train.copy()
    train[TARGET_COL] = y_train
    pipe.fit(train)
    y_test, _ = normalize_target(test[TARGET_COL])
    test = test.copy()
    test[TARGET_COL] = y_test
    Xt = pipe.transform_X(test)
    assert Xt["TLD"].iloc[0] == "OTHER"
    assert "bin_" in Xt["URLLength"].iloc[0]


def test_high_cardinality_columns_not_in_transform():
    cfg = PreprocessConfig(drop_high_card_text=True, n_bins=3, tld_top_n=5)
    pipe = PreprocessingPipeline(config=cfg)
    train = _tiny_train_df()
    train["URL"] = ["http://a", "http://b", "http://c", "http://d"]
    y_train, _ = normalize_target(train[TARGET_COL])
    train = train.copy()
    train[TARGET_COL] = y_train
    pipe.fit(train)
    for c in HIGH_CARD_TEXT_COLS:
        assert c not in pipe.feature_columns
    Xt = pipe.transform_X(train)
    assert not any(c in Xt.columns for c in HIGH_CARD_TEXT_COLS)


def test_train_transform_no_error_on_test_numeric():
    cfg = PreprocessConfig(n_bins=4, tld_top_n=10)
    pipe = PreprocessingPipeline(config=cfg)
    train = _tiny_train_df()
    test = _tiny_test_df()
    train = train.copy()
    test = test.copy()
    train[TARGET_COL], _ = normalize_target(train[TARGET_COL])
    test[TARGET_COL], _ = normalize_target(test[TARGET_COL])
    pipe.fit(train)
    pipe.transform_X(train)
    pipe.transform_X(test)


def test_per_column_categorical_top_values(monkeypatch):
    monkeypatch.setattr(preprocessing, "CATEGORICAL_FEATURES", ("TLD", "region"))

    train = pd.DataFrame(
        {
            TARGET_COL: [0, 0, 1, 1],
            "TLD": ["com", "com", "org", "org"],
            "region": ["eu", "eu", "us", "asia"],
            "URLLength": [1.0, 2.0, 3.0, 4.0],
        }
    )
    cfg = PreprocessConfig(tld_top_n=1, n_bins=3)
    pipe = PreprocessingPipeline(config=cfg)
    train[TARGET_COL], _ = normalize_target(train[TARGET_COL])
    pipe.fit(train)

    assert "TLD" in pipe.categorical_top_values
    assert "region" in pipe.categorical_top_values
    assert pipe.categorical_top_values["TLD"] != pipe.categorical_top_values["region"]
