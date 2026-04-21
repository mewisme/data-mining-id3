from __future__ import annotations

import streamlit as st

SECTION_DEFS: list[tuple[str, str, str]] = [
    ("intro", "Introduction & Dataset Source", "Giới thiệu & nguồn dữ liệu"),
    ("data_overview", "Data Overview", "Tổng quan dữ liệu"),
    ("training", "Training Controls", "Điều khiển huấn luyện"),
    ("preprocess_summary", "Preprocessing Summary", "Tóm tắt tiền xử lý"),
    ("pipeline", "Data Processing Pipeline", "Quy trình xử lý dữ liệu"),
    ("evaluation", "Evaluation & Metrics", "Đánh giá & chỉ số"),
    ("tree_viz", "Decision Tree Visualisation", "Trực quan hoá cây quyết định"),
    ("prediction", "Prediction & Rule Explanation", "Dự đoán & giải thích luật"),
]

PIPELINE_STEPS: list[tuple[int, str, str]] = [
    (1, "Raw Data", "Dữ liệu thô"),
    (2, "Feature Selection", "Chọn đặc trưng"),
    (3, "Train/Test Split", "Chia train/test"),
    (4, "Missing Values", "Giá trị thiếu"),
    (5, "Numeric Binning", "Rời rạc hóa số"),
    (6, "Categorical Processing", "Xử lý phân loại"),
    (7, "Model Input Preview", "Xem trước dữ liệu mô hình"),
]

TRAINING_ARTIFACT_KEYS: tuple[str, ...] = (
    "pipe",
    "model",
    "test_df",
    "train_df",
    "work_df",
    "X_train",
    "X_test",
    "y_test",
    "preprocess_info",
)


def tr(vi: str, en: str, lang: str) -> str:
    if lang == "Tiếng Việt":
        if vi.strip() == en.strip():
            return en
        return vi
    return en


def L(lang: str, en: str, vi: str) -> str:
    return tr(vi, en, lang)


def format_path_rule(path: list[tuple[str, object, str]], lang: str) -> tuple[str, str]:
    conditions: list[str] = []
    predicted_class = "unknown"
    for feat, val, _ in path:
        if feat == "class":
            predicted_class = str(val)
        else:
            conditions.append(f"{feat} = {val}")
    if conditions:
        rule = "IF " + " AND ".join(conditions) + f" THEN class = {predicted_class}"
        cond_txt = ", ".join(conditions)
        if lang == "Tiếng Việt":
            natural = f"Mẫu đi qua các nhánh {cond_txt}, nên mô hình gán nhãn **{predicted_class}**."
        else:
            natural = f"This sample followed the branch where {cond_txt}, so the model classified it as **{predicted_class}**."
    else:
        rule = f"THEN class = {predicted_class}"
        if lang == "Tiếng Việt":
            natural = f"Cây kết thúc ngay tại lá (leaf), mô hình gán nhãn **{predicted_class}**."
        else:
            natural = f"The tree reached a direct leaf, so the model classified it as **{predicted_class}**."
    return rule, natural


def init_ui_state(default_section: str, default_step: int) -> None:
    if "active_section" not in st.session_state:
        st.session_state["active_section"] = default_section
    if "active_pipeline_step" not in st.session_state:
        st.session_state["active_pipeline_step"] = default_step


def pipeline_step_states() -> dict[int, str]:
    trained = has_training_artifacts()
    states: dict[int, str] = {}
    for idx, _, _ in PIPELINE_STEPS:
        if idx <= 2:
            states[idx] = "available"
        elif trained:
            states[idx] = "available"
        else:
            states[idx] = "pending"
    return states


def has_training_artifacts(required_keys: tuple[str, ...] = TRAINING_ARTIFACT_KEYS) -> bool:
    return all(key in st.session_state for key in required_keys)


def render_section(title: str, section_id: str):
    expanded = st.session_state.get("active_section") == section_id
    return st.expander(title, expanded=expanded)


def step_status_label(step_idx: int, step_states: dict[int, str], active_step: int, lang: str) -> str:
    if step_states.get(step_idx) == "pending":
        return L(lang, "Pending", "Đang chờ")
    if step_idx < active_step:
        return L(lang, "Completed", "Hoàn thành")
    if step_idx == active_step:
        return L(lang, "Active", "Đang hoạt động")
    return L(lang, "Available", "Sẵn sàng")


def sidebar_navigation(lang: str, step_states: dict[int, str]) -> None:
    st.markdown("---")
    st.subheader(L(lang, "Navigation", "Điều hướng"))
    for pos, (sec_id, sec_en, sec_vi) in enumerate(SECTION_DEFS, start=1):
        label = L(lang, sec_en, sec_vi)
        is_active = st.session_state.get("active_section") == sec_id
        prefix = "▶" if is_active else "•"
        if st.button(f"{prefix} {pos}. {label}", key=f"nav_{sec_id}", use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state["active_section"] = sec_id

        if sec_id == "pipeline":
            st.caption(L(lang, "Pipeline steps", "Các bước pipeline"))
            for idx, en, vi in PIPELINE_STEPS:
                status = step_status_label(idx, step_states, st.session_state.get("active_pipeline_step", 1), lang)
                st.caption(f"↳ Step {idx}: {L(lang, en, vi)} ({status})")
