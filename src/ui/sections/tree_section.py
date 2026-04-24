"""Decision Tree visualisation section (graph + text rules)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.id3 import ID3Classifier
from src.ui.charts import plot_tree_graphviz
from src.ui.common import L, has_training_artifacts


def render_tree_section(lang: str) -> None:
    """Render the decision tree graph and text-rules tabs."""
    if not has_training_artifacts(("model",)):
        st.caption(L(lang, "Train the model first to view the tree.", "Huấn luyện mô hình trước để xem cây."))
        return

    model: ID3Classifier = st.session_state["model"]
    if model.root_ is None:
        st.caption(L(lang, "Model has no tree root — please retrain.", "Mô hình chưa có gốc cây — hãy train lại."))
        return

    tab_text, tab_graph = st.tabs(
        [
            L(lang, "📋 Text Rules", "📋 Luật dạng text"),
            L(lang, "🌳 Interactive Graph", "🌳 Đồ thị tương tác"),
        ]
    )

    # ── Text rules tab ────────────────────────────────────────────────────────
    with tab_text:
        max_rules = st.slider(
            L(lang, "Max rules to display", "Số luật tối đa hiển thị"),
            min_value=5,
            max_value=200,
            value=32,
            step=5,
        )
        rules = model.rules_to_text(max_rules=max_rules)
        if not rules:
            st.caption(L(lang, "No rules generated.", "Không có luật nào được tạo."))
            return

        st.caption(
            L(
                lang,
                f"Showing {len(rules)} rule(s). Each rule is a root-to-leaf path in the tree.",
                f"Hiển thị {len(rules)} luật. Mỗi luật là một đường từ gốc đến lá trong cây.",
            )
        )

        # Colour-coded display: phishing rules in red-ish, legitimate in green-ish
        phishing_rules = [r for r in rules if "phishing" in r.lower()]
        legit_rules = [r for r in rules if "legitimate" in r.lower()]
        other_rules = [r for r in rules if r not in phishing_rules and r not in legit_rules]

        if phishing_rules:
            st.markdown(f"**🔴 {L(lang, 'Phishing rules', 'Luật phishing')} ({len(phishing_rules)})**")
            st.code("\n".join(phishing_rules), language="text")

        if legit_rules:
            st.markdown(f"**🟢 {L(lang, 'Legitimate rules', 'Luật legitimate')} ({len(legit_rules)})**")
            st.code("\n".join(legit_rules), language="text")

        if other_rules:
            st.markdown(f"**⚪ {L(lang, 'Other rules', 'Luật khác')} ({len(other_rules)})**")
            st.code("\n".join(other_rules), language="text")

    # ── Graph tab ─────────────────────────────────────────────────────────────
    with tab_graph:
        with st.expander(L(lang, "ID3 formulas (entropy & information gain)", "Công thức ID3 (entropy & information gain)"), expanded=False):
            st.markdown(f"**{L(lang, 'Core formulas', 'Công thức cốt lõi')}**")
            st.latex(r"H(S) = -\sum_{i} p_i \log_2(p_i)")
            st.latex(r"Gain(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|}H(S_v)")

            var_rows = [
                {
                    L(lang, "Symbol", "Ký hiệu"): "S",
                    L(lang, "Meaning", "Ý nghĩa"): L(lang, "Current sample set at a node", "Tập mẫu hiện tại tại một nút"),
                },
                {
                    L(lang, "Symbol", "Ký hiệu"): "A",
                    L(lang, "Meaning", "Ý nghĩa"): L(lang, "Candidate feature for splitting", "Đặc trưng ứng viên để chia"),
                },
                {
                    L(lang, "Symbol", "Ký hiệu"): "S_v",
                    L(lang, "Meaning", "Ý nghĩa"): L(lang, "Subset where A takes value v", "Tập con khi A nhận giá trị v"),
                },
                {
                    L(lang, "Symbol", "Ký hiệu"): "p_i",
                    L(lang, "Meaning", "Ý nghĩa"): L(lang, "Class probability at the node", "Xác suất lớp tại nút"),
                },
            ]
            st.dataframe(pd.DataFrame(var_rows), width="stretch")

            if "X_train" in st.session_state and "train_df" in st.session_state:
                X_train_df: pd.DataFrame = st.session_state["X_train"]
                y_train = np.asarray(st.session_state["train_df"]["label"]).astype(int).ravel()
                p_legit = float(np.mean(y_train == 1))
                p_phish = float(np.mean(y_train == 0))
                h_parent = 0.0
                for p in (p_legit, p_phish):
                    if p > 0:
                        h_parent -= p * np.log2(p)

                st.markdown(f"**{L(lang, 'Root node example', 'Ví dụ tại nút gốc')}**")
                st.write(
                    L(
                        lang,
                        f"At root: p(legitimate)={p_legit:.4f}, p(phishing)={p_phish:.4f}",
                        f"Tại nút gốc: p(hợp lệ)={p_legit:.4f}, p(lừa đảo)={p_phish:.4f}",
                    )
                )
                st.latex(
                    rf"H(S_{{root}}) = -({p_legit:.4f})\log_2({p_legit:.4f}) - ({p_phish:.4f})\log_2({p_phish:.4f}) = {h_parent:.4f}"
                )

                gain_rows: list[dict[str, float | str]] = []
                for feat in X_train_df.columns:
                    gain_val = float(model._information_gain(y_train, feat, X_train_df[feat]))
                    gain_rows.append(
                        {
                            L(lang, "Feature", "Đặc trưng"): feat,
                            L(lang, "Gain", "Gain"): gain_val,
                        }
                    )
                gain_df = pd.DataFrame(gain_rows).sort_values(by=L(lang, "Gain", "Gain"), ascending=False).head(12)
                st.caption(
                    L(
                        lang,
                        "Top features by information gain at the root (computed on transformed train data). ID3 picks the highest one.",
                        "Top đặc trưng theo information gain tại nút gốc (tính trên dữ liệu train đã transform). ID3 sẽ chọn đặc trưng cao nhất.",
                    )
                )
                st.dataframe(gain_df, width="stretch")

        with st.container():
            max_depth = st.slider(
                L(lang, "Max display depth", "Độ sâu hiển thị tối đa"),
                min_value=1,
                max_value=10,
                value=4,
                help=L(
                    lang,
                    "Limits how many levels of the tree are rendered. "
                    "Increase to see deeper branches (may become crowded).",
                    "Giới hạn số tầng cây được vẽ. "
                    "Tăng lên để xem nhánh sâu hơn (có thể chật).",
                ),
            )
        try:
            dot = plot_tree_graphviz(model.root_, max_depth=max_depth, lang=lang)
            st.graphviz_chart(dot, width="stretch")
        except RuntimeError as exc:
            st.warning(f"{exc}. {L(lang, 'Install dependencies and restart app.', 'Hãy cài dependency và khởi động lại app.')}")
