from __future__ import annotations

import streamlit as st

from src.data_loader import resolve_default_path
from src.ui.common import L


def render_intro_section(lang: str, source: str) -> None:
    if lang == "Tiếng Việt":
        st.markdown(
            """
**Project:** phân loại nhị phân **phishing** và **legitimate** từ đặc trưng URL / trang web, trên **PhiUSIIL Phishing URL Dataset**, bằng **custom ID3 decision tree**
(**entropy**, **information gain**, **recursive splits** — không dùng `sklearn` **DecisionTreeClassifier**).
"""
        )
    else:
        st.markdown(
            """
**Project:** binary classification into **phishing** vs **legitimate** from URL / page features, using the **PhiUSIIL Phishing URL Dataset** and a **custom ID3 decision tree**
(**entropy**, **information gain**, **recursive splits** — not `sklearn` **DecisionTreeClassifier**).
"""
        )

    st.info(
        L(
            lang,
            "**Model:** **custom ID3 decision tree**. **Splitting criterion:** **entropy** and **information gain**. Not `sklearn.tree.DecisionTreeClassifier`.",
            "**Model:** **custom ID3 decision tree**. **Splitting criterion:** **entropy** và **information gain**. Không dùng `sklearn.tree.DecisionTreeClassifier`.",
        )
    )

    if source == "default":
        st.success(L(lang, f"Using default dataset: `{resolve_default_path()}`", f"Đang dùng dữ liệu mặc định: `{resolve_default_path()}`"))
    elif source == "upload":
        st.success(L(lang, "Using uploaded CSV.", "Đang dùng file CSV đã upload."))
