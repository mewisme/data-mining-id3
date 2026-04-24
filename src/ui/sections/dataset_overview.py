from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui.common import L
from src.utils import FEATURE_COLUMNS, TARGET_COL


def render_dataset_overview_section(df: pd.DataFrame, lang: str) -> None:
    st.markdown(
        """
        <style>
        @media (max-width: 768px) {
            [data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
            }
            [data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    n = len(df)
    col_rows, col_cols, col_schema = st.columns(3)
    col_rows.metric(L(lang, "Total rows", "Tổng số dòng"), f"{n:,}")
    col_cols.metric(L(lang, "Total columns", "Tổng số cột"), len(df.columns))
    feat_count = len([c for c in FEATURE_COLUMNS if c in df.columns])
    col_schema.metric(L(lang, "Schema features (present)", "Số feature theo schema (đang có)"), feat_count)

    st.write(f"**{L(lang, 'Preview', 'Xem trước dữ liệu')}**")
    st.dataframe(df.head(15), width="stretch")

    row_2_left, row_2_right = st.columns(2)
    with row_2_left:
        if TARGET_COL in df.columns:
            st.write(f"**{L(lang, 'Target distribution', 'Phân phối nhãn đích')}** (`label`)")
            vc = df[TARGET_COL].value_counts().sort_index()
            st.bar_chart(vc, width="stretch")
            with st.expander(L(lang, "Details", "Chi tiết"), expanded=False):
                st.write(vc.to_dict())

    with row_2_right:
        st.write(f"**{L(lang, 'Missing values', 'Giá trị thiếu')}** ({L(lang, 'top columns', 'các cột đứng đầu')})")
        miss = df.isna().sum().sort_values(ascending=False)
        st.dataframe(miss[miss > 0].head(25), width="stretch")

    row_3_left, row_3_right = st.columns(2)
    with row_3_left:
        st.write(f"**{L(lang, 'Column dtypes', 'Kiểu dữ liệu cột')}**")
        st.dataframe(df.dtypes.astype(str).to_frame("dtype"), width="stretch")

    with row_3_right:
        dtype_summary = df.dtypes.astype(str).value_counts().rename_axis("dtype").reset_index(name="count")
        st.write(f"**{L(lang, 'Feature type summary', 'Tóm tắt kiểu feature')}**")
        st.dataframe(dtype_summary, width="stretch")
