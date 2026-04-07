from __future__ import annotations

import pandas as pd
import streamlit as st

from src.utils import FEATURE_COLUMNS, TARGET_COL


def render_dataset_overview_section(df: pd.DataFrame) -> None:
    n = len(df)
    st.metric("Total rows", f"{n:,}")
    st.metric("Total columns", len(df.columns))
    feat_count = len([c for c in FEATURE_COLUMNS if c in df.columns])
    st.metric("Schema features (present)", feat_count)
    if TARGET_COL in df.columns:
        st.write("**Target distribution** (`label`)")
        vc = df[TARGET_COL].value_counts().sort_index()
        st.bar_chart(vc)
        st.write(vc.to_dict())
    st.write("**Preview**")
    st.dataframe(df.head(15), width="stretch")
    st.write("**Missing values** (top columns)")
    miss = df.isna().sum().sort_values(ascending=False)
    st.dataframe(miss[miss > 0].head(25), width="stretch")
    st.write("**Column dtypes**")
    st.dataframe(df.dtypes.astype(str).to_frame("dtype"), width="stretch")
    dtype_summary = df.dtypes.astype(str).value_counts().rename_axis("dtype").reset_index(name="count")
    st.write("**Feature type summary**")
    st.dataframe(dtype_summary, width="stretch")
