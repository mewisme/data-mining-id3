from __future__ import annotations

from typing import Any, Protocol, TypedDict, cast, runtime_checkable

import pandas as pd
from sklearn.model_selection import train_test_split

from src.id3 import ID3Classifier
from src.preprocessing import BinStrategy, PreprocessConfig, PreprocessingPipeline, normalize_target, preprocessing_summary
from src.utils import TARGET_COL


class TrainingConfigSnapshot(TypedDict):
    n_bins: int
    bin_strategy: BinStrategy
    categorical_top_n: int
    test_size: float
    max_depth: int
    min_samples_split: int
    row_limit: int
    data_row_count: int
    data_columns_fingerprint: tuple[str, ...]


class SplitInfo(TypedDict):
    rows_total: int
    rows_train: int
    rows_test: int
    test_size: float
    random_state: int


class SamplingInfo(TypedDict):
    row_limit_requested: int
    row_sampling_enabled: bool
    rows_used_for_training_pipeline: int
    rows_original: int
    binning_fit_train_only: bool


class TrainingArtifacts(TypedDict):
    pipe: PreprocessingPipeline
    model: ID3Classifier
    test_df: pd.DataFrame
    train_df: pd.DataFrame
    work_df: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_test: Any
    target_mapping: dict[Any, int]
    target_unique_raw: list[str]
    split_info: SplitInfo
    preprocess_info: dict[str, Any]
    sampling_info: SamplingInfo


@runtime_checkable
class SupportsBinRanges(Protocol):
    def bin_ranges(self) -> dict[str, list[str]]: ...


def training_config_snapshot(
    *,
    n_bins: int,
    bin_strategy: BinStrategy,
    categorical_top_n: int,
    test_size: float,
    max_depth: int,
    min_samples_split: int,
    row_limit: int,
    data_row_count: int,
    data_columns_fingerprint: tuple[str, ...],
) -> TrainingConfigSnapshot:
    return {
        "n_bins": int(n_bins),
        "bin_strategy": bin_strategy,
        "categorical_top_n": int(categorical_top_n),
        "test_size": float(test_size),
        "max_depth": int(max_depth),
        "min_samples_split": int(min_samples_split),
        "row_limit": int(row_limit),
        "data_row_count": int(data_row_count),
        "data_columns_fingerprint": data_columns_fingerprint,
    }


def current_training_config_from_ui(
    df: pd.DataFrame,
    *,
    n_bins: int,
    bin_strategy: BinStrategy,
    categorical_top_n: int,
    test_size: float,
    max_depth: int,
    min_samples_split: int,
    row_limit: int,
) -> TrainingConfigSnapshot:
    return training_config_snapshot(
        n_bins=n_bins,
        bin_strategy=bin_strategy,
        categorical_top_n=categorical_top_n,
        test_size=test_size,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        row_limit=row_limit,
        data_row_count=len(df),
        data_columns_fingerprint=tuple(sorted(df.columns.astype(str))),
    )


def safe_bin_ranges(pipe_obj: SupportsBinRanges | object) -> dict[str, list[str]]:
    if isinstance(pipe_obj, SupportsBinRanges):
        try:
            return pipe_obj.bin_ranges()
        except Exception:
            return {}
    out: dict[str, list[str]] = {}
    numeric_cols = getattr(pipe_obj, "numeric_columns", [])
    discretizers = getattr(pipe_obj, "discretizers", {})
    for col in numeric_cols:
        disc = discretizers.get(col) if isinstance(discretizers, dict) else None
        if disc is None or not hasattr(disc, "bin_edges_"):
            continue
        edges = disc.bin_edges_[0]
        labels: list[str] = []
        for i in range(len(edges) - 1):
            left = float(edges[i])
            right = float(edges[i + 1])
            if i < len(edges) - 2:
                labels.append(f"bin_{i}: [{left:.6g}, {right:.6g})")
            else:
                labels.append(f"bin_{i}: [{left:.6g}, {right:.6g}]")
        out[col] = labels
    return out


def run_training(
    df: pd.DataFrame,
    *,
    n_bins: int,
    bin_strategy: BinStrategy,
    tld_top_n: int,
    test_size: float,
    max_depth: int,
    min_samples_split: int,
    row_limit: int,
) -> TrainingArtifacts:
    work = df.copy()
    raw_target_snapshot = work[TARGET_COL].copy()
    y_norm, target_mapping = normalize_target(work[TARGET_COL])
    work[TARGET_COL] = y_norm

    if row_limit and row_limit > 0:
        work = work.sample(n=min(row_limit, len(work)), random_state=42).reset_index(drop=True)

    if len(work) < 2:
        raise ValueError("Need at least 2 rows to split train/test.")
    if not 0.0 < float(test_size) < 1.0:
        raise ValueError(f"test_size must be between 0 and 1 (exclusive), got {test_size}.")

    class_counts = work[TARGET_COL].value_counts(dropna=False)
    can_stratify = bool(len(class_counts) >= 2 and int(class_counts.min()) >= 2)

    if can_stratify:
        try:
            train_df, test_df = train_test_split(
                work,
                test_size=test_size,
                random_state=42,
                stratify=work[TARGET_COL],
            )
        except ValueError:
            train_df, test_df = train_test_split(work, test_size=test_size, random_state=42)
    else:
        train_df, test_df = train_test_split(work, test_size=test_size, random_state=42)

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Train/test split produced an empty partition. Increase dataset size or adjust test_size/row_limit."
        )

    cfg = PreprocessConfig(
        drop_high_card_text=True,
        n_bins=n_bins,
        bin_strategy=bin_strategy,
        tld_top_n=tld_top_n,
    )
    pipe = PreprocessingPipeline(config=cfg)
    pipe.fit(train_df)
    X_train = pipe.transform_X(train_df)
    y_train = train_df[TARGET_COL].astype(int).values
    X_test = pipe.transform_X(test_df)
    y_test = test_df[TARGET_COL].astype(int).values

    model = ID3Classifier(max_depth=int(max_depth), min_samples_split=int(min_samples_split))
    model.fit(X_train, pd.Series(y_train))

    return cast(
        TrainingArtifacts,
        {
        "pipe": pipe,
        "model": model,
        "test_df": test_df,
        "train_df": train_df,
        "work_df": work,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "target_mapping": target_mapping,
        "target_unique_raw": sorted(pd.unique(raw_target_snapshot.astype(str)).tolist()),
        "split_info": {
            "rows_total": int(len(work)),
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
            "test_size": float(test_size),
            "random_state": 42,
        },
        "preprocess_info": preprocessing_summary(cfg, pipe.feature_columns),
        "sampling_info": {
            "row_limit_requested": int(row_limit),
            "row_sampling_enabled": bool(row_limit and row_limit > 0),
            "rows_used_for_training_pipeline": int(len(work)),
            "rows_original": int(len(df)),
            "binning_fit_train_only": True,
        },
        },
    )
