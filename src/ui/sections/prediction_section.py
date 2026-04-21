from __future__ import annotations

import pandas as pd
import streamlit as st

from src.predictor import format_prediction, predict_manual_artifacts, predict_test_row_artifacts
from src.ui.charts import plot_tree_graph
from src.ui.common import L, format_path_rule, has_training_artifacts
from src.utils import MANUAL_PREDICTION_FEATURES, TARGET_COL, label_to_display


def render_prediction_section(lang: str, training_drift: bool) -> None:
    def render_prediction_details(path: list[tuple[str, object, str]], graph_key_prefix: str) -> None:
        rule_text, natural_text = format_path_rule(path, lang)
        tab_text, tab_graph = st.tabs(
            [
                L(lang, "📋 Rule as text", "📋 Luật dạng text"),
                L(lang, "🌳 Tree graph", "🌳 Đồ thị cây"),
            ]
        )
        with tab_text:
            st.code(rule_text)
            st.markdown(f"**{L(lang, 'Explanation', 'Giải thích')}:** {natural_text}")

        with tab_graph:
            st.caption(L(lang, "Optional visual context for this prediction.", "Ngữ cảnh trực quan tuỳ chọn cho dự đoán này."))
            render_optional_tree_graph(graph_key_prefix)

    def render_optional_tree_graph(graph_key_prefix: str) -> None:
        if model.root_ is None:
            return
        max_graph_depth = st.slider(
            L(lang, "Graph max depth", "Độ sâu tối đa của đồ thị"),
            min_value=1,
            max_value=10,
            value=4,
            key=f"{graph_key_prefix}_graph_depth",
        )
        fig = plot_tree_graph(
            model.root_,
            max_depth=max_graph_depth,
            title=L(
                lang,
                f"Prediction context: ID3 tree (depth <= {max_graph_depth})",
                f"Ngữ cảnh dự đoán: cây ID3 (độ sâu <= {max_graph_depth})",
            ),
        )
        st.plotly_chart(fig, width="stretch")

    if training_drift and "model" in st.session_state:
        st.warning(
            L(
                lang,
                "Predictions and rules below use the **last trained** tree/preprocessor, not the current sliders until you retrain.",
                "Dự đoán và luật phía dưới dùng cây/preprocessor **đã train trước đó**, chưa theo thanh trượt hiện tại cho đến khi train lại.",
            )
        )
    if not has_training_artifacts(("model", "pipe", "test_df")):
        st.caption(L(lang, "Train the model to unlock prediction.", "Huấn luyện mô hình để bật phần dự đoán."))
        return

    pipe = st.session_state["pipe"]
    model = st.session_state["model"]
    test_df: pd.DataFrame = st.session_state["test_df"]
    mode = st.radio(
        L(lang, "Prediction mode", "Chế độ dự đoán"),
        ["pick_test_row", "manual_features"],
        format_func=lambda k: (L(lang, "Pick test row (default)", "Chọn dòng tập test (mặc định)") if k == "pick_test_row" else L(lang, "Manual feature subset", "Nhập tay một phần feature")),
        horizontal=True,
    )
    if mode == "pick_test_row":
        idx = st.number_input(L(lang, "Test row index", "Chỉ số dòng trong tập test"), min_value=0, max_value=len(test_df) - 1, value=0)
        row = test_df.iloc[int(idx)]
        if st.button(L(lang, "Run prediction", "Chạy dự đoán")):
            pred, path, transformed_row = predict_test_row_artifacts(model, pipe, row)
            if TARGET_COL in row.index:
                st.write(f"**{L(lang, 'True label', 'Nhãn đúng')}:** `{int(row[TARGET_COL])}` — **{label_to_display(int(row[TARGET_COL]))}**")
            st.write(f"**{L(lang, 'Prediction', 'Dự đoán')}:** **{format_prediction(pred)}**")
            st.dataframe(transformed_row.to_frame().T, width="stretch")
            for step in path:
                st.write(f"- {step[0]} = {step[1]} → {step[2]}")
            render_prediction_details(path, "pred_test_row")
    else:
        updates: dict = {}
        cols = st.columns(2)
        manual_fields = [f for f in MANUAL_PREDICTION_FEATURES if f in pipe.feature_columns]
        for i, fname in enumerate(manual_fields):
            with cols[i % 2]:
                if fname == "TLD":
                    updates[fname] = st.text_input("TLD (e.g. com, de)", value=str(pipe.default_raw_row.get(fname, "com")))
                else:
                    updates[fname] = st.number_input(fname, value=float(pipe.default_raw_row[fname]), format="%.6f")
        if st.button(L(lang, "Run manual prediction", "Dự đoán thủ công")):
            manual_updates = {k: (str(v).strip() or "com") if k == "TLD" else float(v) for k, v in updates.items()}
            pred, path, raw_manual_row, transformed_row = predict_manual_artifacts(model, pipe, manual_updates)
            st.write(f"**{L(lang, 'Prediction', 'Dự đoán')}:** **{format_prediction(pred)}**")
            st.dataframe(raw_manual_row.to_frame().T, width="stretch")
            st.dataframe(transformed_row.to_frame().T, width="stretch")
            render_prediction_details(path, "pred_manual")
