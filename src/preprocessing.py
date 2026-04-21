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
    # Max frequent categories kept per categorical column on train; remainder -> OTHER
    tld_top_n: int = 50
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
    categorical_top_values: dict[str, list[str]] = field(default_factory=dict)
    numeric_fill_values: dict[str, float] = field(default_factory=dict)
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
        self.imputers = {}
        self.discretizers = {}
        self.categorical_top_values = {}
        self.numeric_fill_values = {}

        X_work = X.copy()
        default_vals: dict[str, Any] = {}

        for c in self.numeric_columns:
            s = pd.to_numeric(X_work[c], errors="coerce")
            imp = SimpleImputer(strategy="median")
            imp.fit(s.values.reshape(-1, 1))
            self.imputers[c] = imp
            filled = imp.transform(s.values.reshape(-1, 1)).ravel().astype(float)
            fill_val = float(np.nanmedian(filled))
            self.numeric_fill_values[c] = fill_val
            default_vals[c] = fill_val
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
                        warnings.filterwarnings(
                            "ignore",
                            message=r"Feature .* is constant and will be replaced with 0\.",
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
                    warnings.filterwarnings(
                        "ignore",
                        message=r"Feature .* is constant and will be replaced with 0\.",
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
            self.categorical_top_values[c] = top
            default_vals[c] = top[0] if top else "__MISSING__"

        self.default_raw_row = pd.Series(default_vals)

        return self

    def transform_debug_stages(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Expose intermediate stages for UI/education/debugging views."""
        if not self.feature_columns:
            raise RuntimeError("Call fit() before transform_debug_stages().")
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Input missing columns: {missing[:5]}...")

        selected = df[self.feature_columns].copy()
        missing_handled = selected.copy()

        for c in self.numeric_columns:
            s = pd.to_numeric(selected[c], errors="coerce")
            imp = self.imputers[c]
            filled = imp.transform(s.values.reshape(-1, 1)).ravel().astype(float)
            missing_handled[c] = pd.Series(filled, index=selected.index)

        for c in self.categorical_columns:
            missing_handled[c] = (
                selected[c]
                .astype(str)
                .replace({"nan": np.nan})
                .fillna("__MISSING__")
            )

        transformed = self.transform_X(df)
        return {
            "selected": selected,
            "missing_handled": missing_handled,
            "transformed": transformed,
        }

    def feature_decisions(self, original_columns: list[str]) -> pd.DataFrame:
        """Return used/dropped status and reason for each input column."""
        rows: list[dict[str, str]] = []
        for col in original_columns:
            status = "used"
            reason = "kept as model feature"
            if col == TARGET_COL:
                status = "dropped"
                reason = "target column"
            elif col == ID_COL_DROP:
                status = "dropped"
                reason = "identifier column"
            elif self.config.drop_high_card_text and col in HIGH_CARD_TEXT_COLS:
                status = "dropped"
                reason = "high-cardinality text"
            rows.append({"column": col, "status": status, "reason": reason})
        return pd.DataFrame(rows)

    def numeric_binning_details(self) -> dict[str, dict[str, Any]]:
        """Return effective bins and edges learned per numeric column."""
        out: dict[str, dict[str, Any]] = {}
        for c in self.numeric_columns:
            disc = self.discretizers.get(c)
            if disc is None or not hasattr(disc, "bin_edges_"):
                continue
            edges = [float(v) for v in disc.bin_edges_[0].tolist()]
            n_bins_eff = len(edges) - 1
            out[c] = {
                "requested_bins": int(self.config.n_bins),
                "effective_bins": int(n_bins_eff),
                "strategy": str(self.config.bin_strategy),
                "edges": edges,
            }
        return out

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
            top = set(self.categorical_top_values.get(c, []))
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


# Canonical string tokens -> 0 (legitimate) / 1 (phishing). Unknown tokens are errors, never defaulted.
_TARGET_STR_TO_BIN: dict[str, int] = {
    "0": 0,
    "legitimate": 0,
    "benign": 0,
    "safe": 0,
    "no": 0,
    "1": 1,
    "phishing": 1,
    "malicious": 1,
    "yes": 1,
}


def normalize_target(y: pd.Series) -> tuple[pd.Series, dict[Any, int]]:
    """Map label column to int 0 (legitimate) / 1 (phishing).

    - Rejects NaN in the target.
    - Accepts numeric labels only if every value is exactly 0 or 1 (including 0.0 / 1.0).
    - Accepts a fixed set of string tokens (case-insensitive); any other string raises ValueError.
    """
    if y.isna().any():
        n_na = int(y.isna().sum())
        raise ValueError(f"Target column contains {n_na} missing value(s); drop or impute them before training.")

    num = pd.to_numeric(y, errors="coerce")
    if num.notna().all():
        bad = num[(num != 0) & (num != 1)]
        if len(bad) > 0:
            samp = sorted(pd.unique(bad))[:10]
            raise ValueError(
                "Target must be binary 0/1 (or equivalent floats). "
                f"Found non-binary numeric value(s), e.g.: {samp}"
            )
        y_out = num.astype(int)
        return y_out, {0: 0, 1: 1}

    lower = y.astype(str).str.lower().str.strip()
    unknown: list[str] = []
    out_vals: list[int] = []
    mapping_raw: dict[str, int] = {}
    for token in pd.unique(lower):
        if token not in _TARGET_STR_TO_BIN:
            unknown.append(token)
        else:
            mapping_raw[token] = _TARGET_STR_TO_BIN[token]
    if unknown:
        show = unknown[:15]
        raise ValueError(
            "Unsupported target label value(s): "
            f"{show!r}. "
            "Use 0/1 or one of: "
            + ", ".join(sorted(_TARGET_STR_TO_BIN))
        )
    y_out = lower.map(mapping_raw)
    return y_out.astype(int), dict(mapping_raw)


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
