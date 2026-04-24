from __future__ import annotations

import pandas as pd
import streamlit as st

from src.evaluation import confusion_matrix_df, evaluate, report_df
from src.id3 import ID3Classifier
from src.ui.common import L, has_training_artifacts


def render_evaluation_section(lang: str, training_drift: bool) -> None:
    _ = training_drift
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

    with st.expander(L(lang, "Metric formulas", "Công thức các chỉ số"), expanded=False):
        st.markdown(f"**1) {L(lang, 'Accuracy', 'Accuracy')}**")
        st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}")
        st.markdown(
            L(
                lang,
                "- `TP`: true phishing predicted as phishing.\n"
                "- `TN`: true legitimate predicted as legitimate.\n"
                "- `FP`: legitimate predicted as phishing.\n"
                "- `FN`: phishing predicted as legitimate.",
                "- `TP`: phishing thật, dự đoán đúng là phishing.\n"
                "- `TN`: hợp lệ thật, dự đoán đúng là hợp lệ.\n"
                "- `FP`: hợp lệ thật nhưng dự đoán thành phishing.\n"
                "- `FN`: phishing thật nhưng dự đoán thành hợp lệ.",
            )
        )

        st.markdown(f"**2) {L(lang, 'Precision', 'Precision')}**")
        st.latex(r"\text{Precision} = \frac{TP}{TP + FP}")
        st.caption(
            L(
                lang,
                "Among samples predicted as phishing, how many are truly phishing.",
                "Trong các mẫu bị dự đoán là phishing, có bao nhiêu mẫu thực sự là phishing.",
            )
        )

        st.markdown(f"**3) {L(lang, 'Recall', 'Recall')}**")
        st.latex(r"\text{Recall} = \frac{TP}{TP + FN}")
        st.caption(
            L(
                lang,
                "Among true phishing samples, how many are correctly detected.",
                "Trong các mẫu phishing thực sự, có bao nhiêu mẫu được phát hiện đúng.",
            )
        )

        st.markdown(f"**4) F1-score**")
        st.latex(r"F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}")
        st.caption(
            L(
                lang,
                "Harmonic mean of Precision and Recall (higher is better when balancing both).",
                "Trung bình điều hòa của Precision và Recall (cao hơn là tốt hơn khi cần cân bằng cả hai).",
            )
        )

    st.write(f"**{L(lang, 'Confusion matrix', 'Ma trận nhầm lẫn')}**")
    st.dataframe(confusion_matrix_df(y_test, y_pred), width="stretch")
    st.write(f"**{L(lang, 'Classification report', 'Báo cáo phân loại')}**")
    with st.expander(L(lang, "How to read each row/column", "Ý nghĩa từng dòng/cột"), expanded=False):
        st.markdown(
            L(
                lang,
                "**Rows**\n"
                "1. `legitimate`: metrics for legitimate class.\n"
                "2. `phishing`: metrics for phishing class.\n"
                "3. `accuracy`: overall correct rate on all samples.\n"
                "4. `macro avg`: unweighted average across classes.\n"
                "5. `weighted avg`: average weighted by class support.\n\n"
                "**Columns**\n"
                "- `precision`: correctness among predicted samples of that row class.\n"
                "- `recall`: detection rate among true samples of that row class.\n"
                "- `f1-score`: harmonic mean of precision and recall.\n"
                "- `support`: number of true samples for that row class.",
                "**Các dòng**\n"
                "1. `legitimate`: chỉ số cho lớp hợp lệ.\n"
                "2. `phishing`: chỉ số cho lớp lừa đảo.\n"
                "3. `accuracy`: tỉ lệ dự đoán đúng trên toàn bộ mẫu.\n"
                "4. `macro avg`: trung bình cộng không trọng số giữa các lớp.\n"
                "5. `weighted avg`: trung bình có trọng số theo số mẫu mỗi lớp.\n\n"
                "**Các cột**\n"
                "- `precision`: độ đúng trong các mẫu được dự đoán là lớp ở dòng đó.\n"
                "- `recall`: tỉ lệ bắt đúng trong các mẫu thật của lớp ở dòng đó.\n"
                "- `f1-score`: trung bình điều hòa giữa precision và recall.\n"
                "- `support`: số mẫu thật của lớp ở dòng đó.",
            )
        )
    st.dataframe(report_df(y_test, y_pred), width="stretch")
