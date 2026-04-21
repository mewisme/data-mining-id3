"""Decision Tree visualisation section (graph + text rules)."""

from __future__ import annotations

import streamlit as st

from src.id3 import ID3Classifier
from src.ui.charts import plot_tree_graph
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
        col_ctrl, col_info = st.columns([2, 3])
        with col_ctrl:
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
        with col_info:
            st.info(
                L(
                    lang,
                    "- 🔵 **Blue circles** = internal split nodes (hover for feature & sample counts).\n"
                    "- 🟢 **Green squares** = leaf predicting **Legitimate**.\n"
                    "- 🔴 **Red squares** = leaf predicting **Phishing**.\n"
                    "- Edge labels show the feature value that leads to that branch.",
                    "- 🔵 **Vòng tròn xanh** = nút rẽ nhánh (hover để xem đặc trưng & số mẫu).\n"
                    "- 🟢 **Hình vuông xanh lá** = lá dự đoán **Legitimate** (hợp lệ).\n"
                    "- 🔴 **Hình vuông đỏ** = lá dự đoán **Phishing**.\n"
                    "- Nhãn cạnh là giá trị đặc trưng dẫn đến nhánh đó.",
                )
            )

        fig = plot_tree_graph(
            model.root_,
            max_depth=max_depth,
            title=L(lang, f"ID3 Decision Tree (depth ≤ {max_depth})", f"Cây quyết định ID3 (độ sâu ≤ {max_depth})"),
        )
        st.plotly_chart(fig, width="stretch")
