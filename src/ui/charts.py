from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ModuleNotFoundError:
    px = None
    go = None

from src.utils import TARGET_COL

if TYPE_CHECKING:
    from src.id3 import ID3Node


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


def plot_tree_graph(root: "ID3Node", max_depth: int = 4, title: str = "ID3 Decision Tree") -> "go.Figure":
    """Render an interactive Plotly graph of the ID3 decision tree.

    Uses BFS to assign layered (x, y) positions and draws edges as line
    traces, then overlays node markers with rich hover text.

    Args:
        root: The root ID3Node of a fitted tree.
        max_depth: Maximum tree depth to render (keeps graph readable).
        title: Figure title string.

    Returns:
        A Plotly Figure with edge traces and a node scatter trace.
    """
    if go is None:
        raise RuntimeError("plotly is required")

    # ── 1. BFS to collect (node, depth, parent_id, edge_label) ──────────────
    NodeRecord = tuple  # (node_id, node, depth, parent_id, edge_label)
    records: list[NodeRecord] = []
    queue: deque[tuple[int, "ID3Node", int, int | None, str]] = deque()
    queue.append((0, root, 0, None, ""))
    next_id = 1

    while queue:
        node_id, node, depth, parent_id, edge_label = queue.popleft()
        records.append((node_id, node, depth, parent_id, edge_label))
        if not node.is_leaf and depth < max_depth:
            for val, child in node.children.items():
                queue.append((next_id, child, depth + 1, node_id, str(val)))
                next_id += 1

    # ── 2. Assign x/y positions (layered layout) ─────────────────────────────
    from collections import defaultdict
    depth_counts: dict[int, int] = defaultdict(int)
    depth_offsets: dict[int, int] = defaultdict(int)

    # First pass: count nodes per depth
    for _, _, depth, _, _ in records:
        depth_counts[depth] += 1

    positions: dict[int, tuple[float, float]] = {}
    for node_id, node, depth, parent_id, edge_label in records:
        count = depth_counts[depth]
        idx = depth_offsets[depth]
        depth_offsets[depth] += 1
        x = (idx - (count - 1) / 2.0)
        y = -depth
        positions[node_id] = (x, y)

    # ── 3. Build edge traces ──────────────────────────────────────────────────
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    mid_x: list[float] = []
    mid_y: list[float] = []
    edge_texts: list[str] = []

    for node_id, node, depth, parent_id, edge_label in records:
        if parent_id is None:
            continue
        px0, py0 = positions[parent_id]
        px1, py1 = positions[node_id]
        edge_x += [px0, px1, None]
        edge_y += [py0, py1, None]
        mid_x.append((px0 + px1) / 2)
        mid_y.append((py0 + py1) / 2)
        edge_texts.append(edge_label)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1.5, color="#94a3b8"),
        hoverinfo="none",
        showlegend=False,
    )

    edge_label_trace = go.Scatter(
        x=mid_x,
        y=mid_y,
        mode="text",
        text=edge_texts,
        textfont=dict(size=9, color="#64748b"),
        hoverinfo="none",
        showlegend=False,
    )

    # ── 4. Build node trace ───────────────────────────────────────────────────
    node_x: list[float] = []
    node_y: list[float] = []
    node_colors: list[str] = []
    node_texts: list[str] = []
    node_hovers: list[str] = []
    node_symbols: list[str] = []

    COLOR_INTERNAL = "#3b82f6"   # blue-500
    COLOR_LEGIT    = "#22c55e"   # green-500
    COLOR_PHISHING = "#ef4444"   # red-500

    for node_id, node, depth, parent_id, edge_label in records:
        px0, py0 = positions[node_id]
        node_x.append(px0)
        node_y.append(py0)

        vc = node.value_counts
        n_leg = vc.get(0, 0)
        n_phi = vc.get(1, 0)
        total = n_leg + n_phi

        if node.is_leaf:
            pred = int(node.prediction if node.prediction is not None else node.majority_label)
            label_str = "Phishing" if pred == 1 else "Legitimate"
            node_colors.append(COLOR_PHISHING if pred == 1 else COLOR_LEGIT)
            node_symbols.append("square")
            node_texts.append(f"🍃 {label_str}")
            node_hovers.append(
                f"<b>Leaf → {label_str}</b><br>"
                f"Samples: {total}<br>"
                f"  Legitimate: {n_leg}<br>"
                f"  Phishing: {n_phi}<br>"
                f"Depth: {depth}"
            )
        else:
            feat = node.feature or "?"
            # Truncate long feature names for the node label
            short = feat if len(feat) <= 18 else feat[:16] + "…"
            children_count = len(node.children)
            node_colors.append(COLOR_INTERNAL)
            node_symbols.append("circle")
            node_texts.append(short)
            node_hovers.append(
                f"<b>Split: {feat}</b><br>"
                f"Samples: {total}<br>"
                f"  Legitimate: {n_leg}<br>"
                f"  Phishing: {n_phi}<br>"
                f"Children: {children_count}<br>"
                f"Depth: {depth}"
            )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=22,
            color=node_colors,
            symbol=node_symbols,
            line=dict(width=2, color="white"),
        ),
        text=node_texts,
        textposition="middle center",
        textfont=dict(size=8, color="white"),
        hovertext=node_hovers,
        hoverinfo="text",
        showlegend=False,
    )

    # ── 5. Legend annotation shapes ───────────────────────────────────────────
    legend_traces = [
        go.Scatter(x=[None], y=[None], mode="markers",
                   marker=dict(size=12, color=COLOR_INTERNAL, symbol="circle"),
                   name="Internal split node"),
        go.Scatter(x=[None], y=[None], mode="markers",
                   marker=dict(size=12, color=COLOR_LEGIT, symbol="square"),
                   name="Leaf: Legitimate"),
        go.Scatter(x=[None], y=[None], mode="markers",
                   marker=dict(size=12, color=COLOR_PHISHING, symbol="square"),
                   name="Leaf: Phishing"),
    ]

    fig = go.Figure(
        data=[edge_trace, edge_label_trace, node_trace] + legend_traces,
        layout=go.Layout(
            title=dict(text=title, font=dict(size=16)),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            margin=dict(l=20, r=20, t=80, b=20),
            plot_bgcolor="#0f172a",
            paper_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"),
        ),
    )
    return fig
