"""Preprocess PhiUSIIL data for ID3: discrete features, train-only discretization."""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer

from src.utils import (
    CATEGORICAL_FEATURES,
    HIGH_CARD_TEXT_COLS,
    ID_COL_DROP,
    TARGET_COL,
)


BinStrategy = Literal["quantile", "uniform"]


@dataclass
class PreprocessConfig:
    drop_high_card_text: bool = True
    tld_top_n: int = 50  # keep top N TLDs; rest -> OTHER
    n_bins: int = 5
    bin_strategy: BinStrategy = "quantile"


@dataclass
class PreprocessingPipeline:
    """Fit on training data only; transform train/test/predict consistently."""

    config: PreprocessConfig = field(default_factory=PreprocessConfig)
    feature_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    imputers: dict[str, SimpleImputer] = field(default_factory=dict)
    discretizers: dict[str, KBinsDiscretizer] = field(default_factory=dict)
    tld_top_categories: list[str] | None = None
    # One raw-feature row (median/mode) for filling manual prediction gaps
    default_raw_row: pd.Series | None = None

    def _select_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if TARGET_COL not in df.columns:
            raise ValueError(f"Missing '{TARGET_COL}'")
        y = df[TARGET_COL].copy()
        X = df.drop(columns=[TARGET_COL], errors="ignore")
        if ID_COL_DROP in X.columns:
            X = X.drop(columns=[ID_COL_DROP])
        if self.config.drop_high_card_text:
            to_drop = [c for c in HIGH_CARD_TEXT_COLS if c in X.columns]
            if to_drop:
                X = X.drop(columns=to_drop)
        self.feature_columns = list(X.columns)
        return X, y

    def _infer_numeric_categorical(self, X: pd.DataFrame) -> None:
        self.numeric_columns = []
        self.categorical_columns = []
        for c in X.columns:
            if c in CATEGORICAL_FEATURES:
                self.categorical_columns.append(c)
            else:
                self.numeric_columns.append(c)

    def fit(self, df: pd.DataFrame) -> "PreprocessingPipeline":
        X, _ = self._select_features(df)
        self._infer_numeric_categorical(X)

        X_work = X.copy()
        default_vals: dict[str, Any] = {}

        for c in self.numeric_columns:
            s = pd.to_numeric(X_work[c], errors="coerce")
            imp = SimpleImputer(strategy="median")
            imp.fit(s.values.reshape(-1, 1))
            self.imputers[c] = imp
            filled = imp.transform(s.values.reshape(-1, 1)).ravel().astype(float)
            default_vals[c] = float(np.nanmedian(filled))
            n_unique = len(np.unique(filled))
            n_bins = max(2, min(self.config.n_bins, n_unique))
            last_err: Exception | None = None
            disc: KBinsDiscretizer | None = None
            kbin_params = inspect.signature(KBinsDiscretizer.__init__).parameters
            extra_kw: dict[str, Any] = {}
            if "quantile_method" in kbin_params:
                extra_kw["quantile_method"] = "averaged_inverted_cdf"
            for try_nb in range(n_bins, 1, -1):
                disc = KBinsDiscretizer(
                    n_bins=try_nb,
                    encode="ordinal",
                    strategy=self.config.bin_strategy,
                    subsample=min(200_000, len(filled)),
                    **extra_kw,
                )
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=r"Bins whose width are too small.*",
                            category=UserWarning,
                            module=r"sklearn\.preprocessing\._discretization",
                        )
                        disc.fit(filled.reshape(-1, 1))
                    last_err = None
                    break
                except ValueError as e:
                    last_err = e
                    continue
            if last_err is not None:
                disc = KBinsDiscretizer(
                    n_bins=2,
                    encode="ordinal",
                    strategy="uniform",
                    subsample=min(200_000, len(filled)),
                    **extra_kw,
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"Bins whose width are too small.*",
                        category=UserWarning,
                        module=r"sklearn\.preprocessing\._discretization",
                    )
                    disc.fit(filled.reshape(-1, 1))
            assert disc is not None
            self.discretizers[c] = disc

        for c in self.categorical_columns:
            vc = (
                X_work[c]
                .astype(str)
                .replace({"nan": np.nan})
                .fillna("__MISSING__")
                .value_counts()
            )
            top = vc.head(self.config.tld_top_n).index.astype(str).tolist()
            self.tld_top_categories = top
            default_vals[c] = top[0] if top else "__MISSING__"

        self.default_raw_row = pd.Series(default_vals)

        return self

    def transform_X(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return fully discrete string DataFrame for ID3 (no target)."""
        if not self.feature_columns:
            raise RuntimeError("Call fit() before transform_X().")
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Input missing columns: {missing[:5]}...")

        X = df[self.feature_columns].copy()
        out: dict[str, pd.Series] = {}

        for c in self.numeric_columns:
            s = pd.to_numeric(X[c], errors="coerce")
            imp = self.imputers[c]
            filled = imp.transform(s.values.reshape(-1, 1)).ravel().astype(float)
            disc = self.discretizers[c]
            try:
                bin_idx = disc.transform(filled.reshape(-1, 1)).ravel().astype(int)
            except ValueError:
                # fallback: clip to valid range
                bin_idx = disc.transform(np.clip(filled, filled.min(), filled.max()).reshape(-1, 1)).ravel().astype(int)
            out[c] = pd.Series([f"bin_{int(b)}" for b in bin_idx], index=X.index)

        for c in self.categorical_columns:
            raw = X[c].astype(str).replace({"nan": np.nan}).fillna("__MISSING__")
            top = set(self.tld_top_categories or [])
            out[c] = raw.where(raw.isin(top), other="OTHER")

        return pd.DataFrame(out, index=X.index)

    def manual_input_frame(self, updates: dict[str, Any]) -> pd.DataFrame:
        """Single row combining training defaults with user-provided feature values."""
        if self.default_raw_row is None or not self.feature_columns:
            raise RuntimeError("Call fit() before manual_input_frame().")
        row = self.default_raw_row.copy()
        for k, v in updates.items():
            if k in row.index:
                row[k] = v
        return pd.DataFrame([row])

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X, y = self._select_features(df)
        self.fit(df)
        Xd = self.transform_X(df)
        return Xd, y

    def bin_ranges(self) -> dict[str, list[str]]:
        """Human-readable bin ranges for each numeric feature."""
        out: dict[str, list[str]] = {}
        for col in self.numeric_columns:
            disc = self.discretizers.get(col)
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


def normalize_target(y: pd.Series) -> tuple[pd.Series, dict[Any, int]]:
    """Map label column to int 0/1 (legitimate/phishing). Dataset is usually already 0/1."""
    uniq = sorted(pd.unique(y.dropna()))
    mapping: dict[Any, int] = {}
    if set(uniq) <= {0, 1}:
        y_out = y.astype(int)
        return y_out, {0: 0, 1: 1}
    # heuristic string labels
    lower = y.astype(str).str.lower().str.strip()
    for u in pd.unique(lower.dropna()):
        if u in ("0", "legitimate", "benign", "safe", "no"):
            mapping[u] = 0
        elif u in ("1", "phishing", "malicious", "yes"):
            mapping[u] = 1
    if len(mapping) >= 2:
        y_out = lower.map(lambda v: mapping.get(v, 0))
        return y_out.astype(int), mapping
    raise ValueError(f"Cannot normalize target; unique values: {uniq[:20]}")


def preprocessing_summary(config: PreprocessConfig, feature_columns: list[str]) -> dict[str, Any]:
    return {
        "dropped_identifier": ID_COL_DROP,
        "target": TARGET_COL,
        "dropped_high_card_text_default": list(HIGH_CARD_TEXT_COLS) if config.drop_high_card_text else [],
        "categorical_kept": [c for c in CATEGORICAL_FEATURES if c in feature_columns],
        "numeric_discretized": [c for c in feature_columns if c not in CATEGORICAL_FEATURES],
        "n_bins": config.n_bins,
        "bin_strategy": config.bin_strategy,
        "tld_top_n": config.tld_top_n,
    }
