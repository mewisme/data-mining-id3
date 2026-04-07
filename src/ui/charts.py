from __future__ import annotations

import pandas as pd
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ModuleNotFoundError:
    px = None
    go = None

from src.utils import TARGET_COL


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None, barmode: str = "group") -> go.Figure:
    if px is None:
        raise RuntimeError("plotly is required")
    fig = px.bar(df, x=x, y=y, color=color, barmode=barmode, title=title)
    fig.update_layout(height=340, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_split_distribution(work_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> go.Figure:
    full_counts = work_df[TARGET_COL].value_counts().sort_index()
    train_counts = train_df[TARGET_COL].value_counts().sort_index()
    test_counts = test_df[TARGET_COL].value_counts().sort_index()
    labels = sorted(set(full_counts.index).union(set(train_counts.index)).union(set(test_counts.index)))
    rows: list[dict[str, object]] = []
    for lab in labels:
        rows.append({"split": "full", "label": str(lab), "count": int(full_counts.get(lab, 0))})
        rows.append({"split": "train", "label": str(lab), "count": int(train_counts.get(lab, 0))})
        rows.append({"split": "test", "label": str(lab), "count": int(test_counts.get(lab, 0))})
    return plot_bar(pd.DataFrame(rows), x="split", y="count", color="label", title="Class balance: full vs train vs test")


def plot_numeric_before_after(train_df: pd.DataFrame, stage: dict[str, pd.DataFrame], col: str, edges: list[float]) -> go.Figure:
    if go is None:
        raise RuntimeError("plotly is required")
    imputed_vals = pd.to_numeric(stage["missing_handled"][col], errors="coerce").dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=imputed_vals, name="imputed_raw", opacity=0.7, nbinsx=30, yaxis="y"))
    edge_y = max(1, len(imputed_vals) // 10)
    for e in edges[1:-1]:
        fig.add_trace(go.Scatter(x=[e, e], y=[0, edge_y], mode="lines", name=f"edge {e:.4g}", line=dict(dash="dot")))
    fig.update_layout(
        title=f"{col}: raw/imputed distribution with bin edges",
        barmode="overlay",
        height=330,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def plot_bin_counts(bin_series: pd.Series, title: str) -> go.Figure:
    counts = bin_series.astype(str).value_counts().sort_index().rename_axis("bin").reset_index(name="count")
    return plot_bar(counts, x="bin", y="count", title=title)


def plot_before_after_category(raw: pd.Series, transformed: pd.Series, top_n: int = 12) -> go.Figure:
    raw_counts = raw.astype(str).value_counts().head(top_n)
    transformed_counts = transformed.astype(str).value_counts().head(top_n)
    out = pd.DataFrame(
        {
            "category": list(raw_counts.index) + list(transformed_counts.index),
            "count": list(raw_counts.values) + list(transformed_counts.values),
            "stage": ["before"] * len(raw_counts) + ["after"] * len(transformed_counts),
        }
    )
    return plot_bar(out, x="category", y="count", color="stage", title="Category frequency before vs after")


def plot_feature_space(df_plot: pd.DataFrame, x: str, y: str, z: str | None, color: str, row_idx: int | None = None) -> go.Figure:
    if px is None or go is None:
        raise RuntimeError("plotly is required")
    if z:
        fig = px.scatter_3d(df_plot, x=x, y=y, z=z, color=color, opacity=0.6, title="Feature space explorer (3D)")
        if row_idx is not None and row_idx in df_plot.index:
            r = df_plot.loc[row_idx]
            fig.add_trace(
                go.Scatter3d(
                    x=[r[x]],
                    y=[r[y]],
                    z=[r[z]],
                    mode="markers",
                    marker=dict(size=8, symbol="diamond", color="black"),
                    name="selected_sample",
                )
            )
    else:
        fig = px.scatter(df_plot, x=x, y=y, color=color, opacity=0.65, title="Feature space explorer (2D)")
        if row_idx is not None and row_idx in df_plot.index:
            r = df_plot.loc[row_idx]
            fig.add_trace(
                go.Scatter(
                    x=[r[x]],
                    y=[r[y]],
                    mode="markers",
                    marker=dict(size=12, symbol="diamond", color="black"),
                    name="selected_sample",
                )
            )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
    return fig
