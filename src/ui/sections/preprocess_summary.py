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
            f"- **Train snapshot:** rows used `{sample_info.get('rows_used_for_training_pipeline', 'N/A')}` / original `{sample_info.get('rows_original', 'N/A')}`\n"
            f"- **Sampling enabled:** `{sample_info.get('row_sampling_enabled', False)}`\n"
            f"- **Numeric discretization used:** `{info['bin_strategy']}`, bins=`{info['n_bins']}`\n"
            f"- **Categorical columns in pipeline:** "
            + (", ".join(f"`{c}`" for c in info["categorical_kept"]) or "—")
            + "\n"
            f"- **Binning fit on train only:** `{sample_info.get('binning_fit_train_only', True)}`",
            f"- **Ảnh chụp lần train:** số dòng dùng `{sample_info.get('rows_used_for_training_pipeline', 'N/A')}` / tổng gốc `{sample_info.get('rows_original', 'N/A')}`\n"
            f"- **Có lấy mẫu:** `{sample_info.get('row_sampling_enabled', False)}`\n"
            f"- **Rời rạc hóa số đã dùng:** `{info['bin_strategy']}`, bins=`{info['n_bins']}`\n"
            f"- **Cột phân loại trong pipeline:** "
            + (", ".join(f"`{c}`" for c in info["categorical_kept"]) or "—")
            + "\n"
            f"- **Binning chỉ fit trên train:** `{sample_info.get('binning_fit_train_only', True)}`",
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
