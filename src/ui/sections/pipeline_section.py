from __future__ import annotations

import pandas as pd
import streamlit as st
from src.preprocessing import PreprocessConfig, PreprocessingPipeline
from src.ui.sections.training_controls import TrainingUiParams
from src.ui.charts import (
    plot_bar,
    plot_before_after_category,
    plot_bin_counts,
    plot_feature_space,
    plot_numeric_before_after,
    plot_split_distribution,
)
from src.ui.common import L
from src.utils import TARGET_COL


def _count_missing_like(s: pd.Series) -> int:
    as_str = s.astype(str).str.strip().str.lower()
    return int(s.isna().sum() + ((~s.isna()) & (as_str == "nan")).sum())


def _render_raw_data_section(df: pd.DataFrame, lang: str) -> None:
    st.subheader(L(lang, "1) Raw Data Overview", "1) Tổng quan dữ liệu thô"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Target", TARGET_COL if TARGET_COL in df.columns else "missing")
    if TARGET_COL in df.columns:
        cls = df[TARGET_COL].value_counts(dropna=False).rename_axis("label").reset_index(name="count")
        st.plotly_chart(plot_bar(cls, x="label", y="count", title="Class distribution (raw target)"), width="stretch")


def render_pipeline_section(df: pd.DataFrame, lang: str, params: TrainingUiParams) -> None:
    _render_raw_data_section(df, lang)

    st.subheader(L(lang, "2) Feature Selection / Dropped Columns", "2) Chọn feature / cột bị loại"))
    demo_pipe = PreprocessingPipeline(
        config=PreprocessConfig(
            drop_high_card_text=True,
            n_bins=params["n_bins"],
            bin_strategy=params["bin_strategy"],
            tld_top_n=params["tld_top_n"],
        )
    )
    st.dataframe(demo_pipe.feature_decisions([str(c) for c in df.columns]), width="stretch")

    st.subheader(L(lang, "4) Train/Test Split Visualization", "4) Trực quan train/test split"))
    if "split_info" in st.session_state and "train_df" in st.session_state and "test_df" in st.session_state:
        split_info = st.session_state["split_info"]
        train_df: pd.DataFrame = st.session_state["train_df"]
        test_df: pd.DataFrame = st.session_state["test_df"]
        c_left, c_right = st.columns(2)
        with c_left:
            size_df = pd.DataFrame({"split": ["train", "test"], "rows": [split_info["rows_train"], split_info["rows_test"]]})
            st.plotly_chart(plot_bar(size_df, x="split", y="rows", title="Train vs test size"), width="stretch")
        with c_right:
            st.plotly_chart(plot_split_distribution(st.session_state["work_df"], train_df, test_df), width="stretch")

    st.subheader(L(lang, "6) Numeric Feature Binning / Discretization", "6) Rời rạc hóa cột số"))
    if "pipe" in st.session_state and "train_df" in st.session_state:
        pipe: PreprocessingPipeline = st.session_state["pipe"]
        train_df = st.session_state["train_df"]
        bin_details = pipe.numeric_binning_details()
        if bin_details:
            col = st.selectbox("Numeric column for value -> bin example", options=sorted(bin_details.keys()))
            full_stage = pipe.transform_debug_stages(train_df)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_numeric_before_after(train_df, full_stage, col, bin_details[col]["edges"]), width="stretch")
            with c2:
                st.plotly_chart(plot_bin_counts(full_stage["transformed"][col], f"{col}: bin frequencies"), width="stretch")

    st.subheader(L(lang, "7) Categorical Transformation", "7) Biến đổi cột phân loại"))
    if "pipe" in st.session_state and "train_df" in st.session_state:
        pipe = st.session_state["pipe"]
        train_df = st.session_state["train_df"]
        if pipe.categorical_columns:
            cat_col = st.selectbox("Categorical column example", options=pipe.categorical_columns)
            full_cat_stage = pipe.transform_debug_stages(train_df)
            st.plotly_chart(plot_before_after_category(train_df[cat_col], full_cat_stage["transformed"][cat_col]), width="stretch")

    st.subheader(L(lang, "8) Final Model Input Preview", "8) Xem trước dữ liệu vào mô hình"))
    if "X_train" in st.session_state and "X_test" in st.session_state:
        X_train_df: pd.DataFrame = st.session_state["X_train"]
        feat_pick = st.selectbox("Transformed feature distribution", options=list(X_train_df.columns), key="transformed_feature_pick")
        freq_df = X_train_df[feat_pick].astype(str).value_counts().rename_axis("token").reset_index(name="count")
        st.plotly_chart(plot_bar(freq_df, x="token", y="count", title=f"X_train token frequency: {feat_pick}"), width="stretch")

    st.subheader(L(lang, "9) Feature Space Explorer (2D / 3D)", "9) Khám phá không gian đặc trưng (2D / 3D)"))
    if "work_df" in st.session_state and "pipe" in st.session_state:
        work_df: pd.DataFrame = st.session_state["work_df"]
        pipe: PreprocessingPipeline = st.session_state["pipe"]
        mode_space = st.radio("Feature space mode", ["raw_numeric", "transformed"], horizontal=True)
        if mode_space == "raw_numeric":
            plot_df = work_df.copy()
            numeric_opts = [c for c in pipe.numeric_columns if c in plot_df.columns]
            for c in numeric_opts:
                plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")
        else:
            transformed_full = pipe.transform_X(work_df)
            plot_df = transformed_full.copy()
            numeric_opts = []
            for c in transformed_full.columns:
                plot_df[c] = pd.factorize(transformed_full[c].astype(str))[0]
                numeric_opts.append(c)
        if len(numeric_opts) >= 2:
            c1, c2, c3 = st.columns(3)
            x_feat = c1.selectbox("X feature", options=numeric_opts, index=0)
            y_feat = c2.selectbox("Y feature", options=numeric_opts, index=1)
            use_3d = c3.checkbox("Enable 3D", value=False)
            z_feat = st.selectbox("Z feature (3D)", options=numeric_opts, index=min(2, len(numeric_opts) - 1)) if use_3d else None
            plot_df["class_label"] = work_df[TARGET_COL].astype(str).values if TARGET_COL in work_df.columns else "unknown"
            st.plotly_chart(plot_feature_space(plot_df, x=x_feat, y=y_feat, z=z_feat, color="class_label"), width="stretch")
