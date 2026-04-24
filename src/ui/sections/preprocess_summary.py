from __future__ import annotations

import pandas as pd
import streamlit as st

from src.services.training_service import safe_bin_ranges
from src.ui.common import L, has_training_artifacts


def render_preprocess_summary_section(lang: str) -> None:
    st.subheader(L(lang, "Preprocessing summary (last train)", "Tóm tắt preprocessing (train gần nhất)"))
    if not has_training_artifacts(("preprocess_info",)):
        st.caption(L(lang, "Train once to populate this summary.", "Huấn luyện một lần để hiển thị phần này."))
        return

    info = st.session_state["preprocess_info"]
    sample_info = st.session_state.get("sampling_info", {})
    pipe_for_bins = st.session_state.get("pipe")
    st.markdown(
        L(
            lang,
            f"- **Dropped identifier:** `{info['dropped_identifier']}`\n"
            f"- **Target:** `{info['target']}`\n"
            f"- **Dropped high-cardinality text (always for this demo):** "
            + (", ".join(f"`{c}`" for c in info["dropped_high_card_text_default"]) or "—")
            + "\n"
            f"- **Kept categorical:** "
            + (", ".join(f"`{c}`" for c in info["categorical_kept"]) or "—")
            + "\n"
            f"- **Numeric discretization:** `{info['bin_strategy']}`, bins=`{info['n_bins']}`\n"
            f"- **Binning fit on train only:** `{sample_info.get('binning_fit_train_only', True)}`\n"
            f"- **Row limit / sampling:** `{sample_info.get('row_sampling_enabled', False)}` "
            f"(rows used: {sample_info.get('rows_used_for_training_pipeline', 'N/A')} / {sample_info.get('rows_original', 'N/A')})",
            f"- **Cột định danh bị loại:** `{info['dropped_identifier']}`\n"
            f"- **Nhãn đích:** `{info['target']}`\n"
            f"- **Text có độ phân biệt cao bị loại (mặc định cho demo):** "
            + (", ".join(f"`{c}`" for c in info["dropped_high_card_text_default"]) or "—")
            + "\n"
            f"- **Cột phân loại được giữ:** "
            + (", ".join(f"`{c}`" for c in info["categorical_kept"]) or "—")
            + "\n"
            f"- **Rời rạc hóa cột số:** `{info['bin_strategy']}`, bins=`{info['n_bins']}`\n"
            f"- **Binning chỉ fit trên train:** `{sample_info.get('binning_fit_train_only', True)}`\n"
            f"- **Giới hạn dòng / lấy mẫu:** `{sample_info.get('row_sampling_enabled', False)}` "
            f"(số dòng đã dùng: {sample_info.get('rows_used_for_training_pipeline', 'N/A')} / {sample_info.get('rows_original', 'N/A')})",
        )
    )
    if pipe_for_bins is None:
        return
    st.caption(
        L(
            lang,
            "Edges are fit on **train**. A value landing in `bin_k` follows branch `bin_k` in ID3.",
            "Ngưỡng (**edges**) fit trên **train**. Giá trị rơi vào `bin_k` → nhánh `bin_k` trong ID3.",
        )
    )
    bin_map = safe_bin_ranges(pipe_for_bins)
    if not bin_map:
        st.write(L(lang, "No numeric features to show bin ranges.", "Không có cột numeric để hiển thị bin."))
    else:
        rows: list[dict[str, str]] = []
        for feature in sorted(bin_map.keys()):
            ranges = bin_map[feature]
            for idx, range_text in enumerate(ranges):
                rows.append(
                    {
                        L(lang, "Feature", "Đặc trưng"): feature,
                        L(lang, "Bin", "Bin"): f"bin_{idx}",
                        L(lang, "Range", "Khoảng giá trị"): range_text,
                    }
                )

        st.write(L(lang, "All numeric bin ranges", "Toàn bộ khoảng bin của cột số"))
        st.dataframe(pd.DataFrame(rows), width="stretch")
