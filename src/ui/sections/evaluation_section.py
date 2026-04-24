from __future__ import annotations

import pandas as pd
import streamlit as st

from src.evaluation import confusion_matrix_df, evaluate, report_df
from src.id3 import ID3Classifier
from src.ui.common import L, has_training_artifacts


def render_evaluation_section(lang: str, training_drift: bool) -> None:
    if training_drift:
        st.warning(
            L(
                lang,
                "Training settings or the loaded dataset changed since the last Train. **Metrics below still reflect the last trained model** — click **Train ID3** to refresh them.",
                "Tham số hoặc bộ dữ liệu đã đổi so với lần Train trước. **Metrics bên dưới vẫn là của mô hình train gần nhất** — hãy bấm **Huấn luyện ID3** để cập nhật.",
            )
        )
    if not has_training_artifacts(("model", "X_test", "y_test")):
        st.caption(L(lang, "Train the model first.", "Hãy huấn luyện mô hình trước."))
        return

    model: ID3Classifier = st.session_state["model"]
    X_test: pd.DataFrame = st.session_state["X_test"]
    y_test = st.session_state["y_test"]
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(L(lang, "Accuracy", "Độ chính xác"), f"{metrics['accuracy']:.4f}")
    m2.metric(L(lang, "Precision (phishing)", "Precision (phishing)"), f"{metrics['precision']:.4f}")
    m3.metric(L(lang, "Recall (phishing)", "Recall (phishing)"), f"{metrics['recall']:.4f}")
    m4.metric(L(lang, "F1-score", "Điểm F1"), f"{metrics['f1']:.4f}")
    st.write(f"**{L(lang, 'Confusion matrix', 'Ma trận nhầm lẫn')}**")
    st.dataframe(confusion_matrix_df(y_test, y_pred), width="stretch")
    st.write(f"**{L(lang, 'Classification report', 'Báo cáo phân loại')}**")
    st.dataframe(report_df(y_test, y_pred), width="stretch")
