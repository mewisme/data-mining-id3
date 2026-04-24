from __future__ import annotations

from typing import TypedDict, cast

import pandas as pd
import streamlit as st

from src.config import settings
from src.preprocessing import BinStrategy
from src.services.training_service import current_training_config_from_ui
from src.ui.common import L
from src.utils import ID_COL_DROP, TARGET_COL


class TrainingUiParams(TypedDict):
    n_bins: int
    bin_strategy: BinStrategy
    tld_top_n: int
    test_size: float
    max_depth: int
    min_samples_split: int
    row_limit: int


def render_training_controls_section(df: pd.DataFrame, lang: str) -> tuple[TrainingUiParams, bool, bool]:
    training_drift = False
    st.info(
        L(
            lang,
            "Columns `URL`, `Domain`, and `Title` are **always dropped**: they are almost unique per row, so keeping them would make the custom ID3 tree huge, and numeric coercion would silently ruin those fields. Categorical columns (e.g. `TLD`) use **top-N** on the train split with `OTHER` for rare/unseen values.",
            "Các cột `URL`, `Domain`, `Title` **luôn bị bỏ**: gần như khác nhau mỗi dòng nên giữ lại sẽ làm cây ID3 phình to; ép kiểu số cũng làm hỏng ngữ nghĩa. Cột phân loại (vd. `TLD`) dùng **top-N** trên tập train và `OTHER` cho giá trị hiếm/chưa gặp.",
        )
    )
    n_bins = st.slider(L(lang, "Number of bins (numeric discretization)", "Số **bins** (rời rạc hóa số)"), 3, 15, 5)
    bin_strategy = cast(
        BinStrategy,
        st.selectbox(L(lang, "Bin strategy", "Chiến lược **bin**"), ["quantile", "uniform"], index=0),
    )
    tld_top_n = st.slider(
        L(lang, "Top-N frequent categories per categorical column (else OTHER)", "**Top-N** hạng mục thường gặp / mỗi cột phân loại (còn lại **OTHER**)"),
        10,
        200,
        50,
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        test_size = st.slider(L(lang, "Test split ratio", "Tỉ lệ **test split**"), 0.1, 0.4, 0.2, 0.05)
    with col2:
        max_depth = st.number_input(L(lang, "Max tree depth", "Độ sâu tối đa (**max depth**)"), min_value=1, max_value=50, value=12)
    with col3:
        min_samples_split = st.number_input(L(lang, "Min samples to split", "Ngưỡng **min samples split**"), min_value=2, max_value=5000, value=100)
    with col4:
        row_limit = st.number_input(
            L(lang, "Row limit (`0` = all rows)", "Giới hạn dòng (`0` = full data)"),
            min_value=0,
            max_value=500_000,
            value=settings.default_row_limit,
            step=500,
        )

    if "model" in st.session_state and "training_config" in st.session_state:
        current_cfg = current_training_config_from_ui(
            df,
            n_bins=n_bins,
            bin_strategy=bin_strategy,
            categorical_top_n=tld_top_n,
            test_size=test_size,
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            row_limit=int(row_limit),
        )
        training_drift = current_cfg != st.session_state["training_config"]

    st.markdown(
        L(
            lang,
            f"- **Dropped identifier:** `{ID_COL_DROP}`\n"
            f"- **Target:** `{TARGET_COL}` (`1` = legitimate, `0` = phishing; unsupported or missing labels error out)\n"
            f"- **Always dropped (high-cardinality text):** `URL`, `Domain`, `Title`\n"
            f"- **Categorical columns:** top-{tld_top_n} per column on **train** + `OTHER`\n"
            f"- **Numeric discretization:** `{bin_strategy}`, bins=`{n_bins}`, fit on **train** only, applied to **test** / predict",
            f"- **Cột định danh bị loại:** `{ID_COL_DROP}`\n"
            f"- **Nhãn đích:** `{TARGET_COL}` (`1` = hợp lệ, `0` = phishing; nhãn thiếu hoặc không hỗ trợ sẽ báo lỗi)\n"
            f"- **Luôn loại bỏ (text có độ phân biệt cao):** `URL`, `Domain`, `Title`\n"
            f"- **Cột phân loại:** top-{tld_top_n} cho mỗi cột trên **train** + `OTHER`\n"
            f"- **Rời rạc hóa số:** `{bin_strategy}`, bins=`{n_bins}`, fit **chỉ trên train**, áp dụng cho **test** / dự đoán",
        )
    )

    params: TrainingUiParams = {
        "n_bins": int(n_bins),
        "bin_strategy": bin_strategy,
        "tld_top_n": int(tld_top_n),
        "test_size": float(test_size),
        "max_depth": int(max_depth),
        "min_samples_split": int(min_samples_split),
        "row_limit": int(row_limit),
    }
    train_btn = st.button(L(lang, "Train ID3", "Huấn luyện **ID3**"), type="primary")
    return params, training_drift, train_btn
