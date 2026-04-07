"""
Streamlit UI: PhiUSIIL phishing URL classification with custom ID3.
Run: streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ModuleNotFoundError:
    px = None
    go = None

from src.config import settings
from src.data_loader import resolve_default_path, validate_schema
from src.services.data_service import load_dataframe
from src.services.training_service import current_training_config_from_ui, run_training
from src.ui.common import L, PIPELINE_STEPS, init_ui_state, pipeline_step_states, render_section
from src.ui.sections.dataset_overview import render_dataset_overview_section
from src.ui.sections.evaluation_section import render_evaluation_section
from src.ui.sections.intro import render_intro_section
from src.ui.sections.pipeline_section import render_pipeline_section
from src.ui.sections.prediction_section import render_prediction_section
from src.ui.sections.preprocess_summary import render_preprocess_summary_section
from src.ui.sections.training_controls import TrainingUiParams, render_training_controls_section
from src.utils import project_root


def main() -> None:
    st.set_page_config(page_title="ID3 Phishing URL Classifier", layout="wide")
    if px is None or go is None:
        st.error("Missing dependency: plotly. Install with `pip install -r requirements.txt` and restart Streamlit.")
        st.stop()

    with st.sidebar:
        lang = st.selectbox("Language / Ngôn ngữ", ["English", "Tiếng Việt"], index=0)
        st.header(L(lang, "Data source", "Nguồn dữ liệu"))
        uploaded = st.file_uploader(L(lang, "PhiUSIIL CSV upload", "Tải lên CSV PhiUSIIL"), type=["csv"])
        default_path = resolve_default_path()
        st.caption(
            L(
                lang,
                f"Default dataset found: `{default_path}`. Loaded automatically if you do not upload a file." if default_path.is_file() else "No default CSV in `data/`. Upload a PhiUSIIL CSV to continue.",
                f"Đã có file mặc định `{default_path}` — ưu tiên dùng nếu bạn không upload file khác." if default_path.is_file() else "Không thấy CSV mặc định trong `data/`. Hãy upload file PhiUSIIL.",
            )
        )

    step_states = pipeline_step_states()
    default_step = min([idx for idx, _en, _vi in PIPELINE_STEPS if step_states.get(idx) != "pending"], default=1)
    init_ui_state(default_section="data_overview", default_step=default_step)

    st.title(L(lang, "Phishing URL classification with ID3 (PhiUSIIL)", "Phân loại URL phishing bằng ID3 (PhiUSIIL)"))

    data_result = load_dataframe(uploaded)
    df = data_result["dataframe"]
    source = data_result["source"]
    if source == "error":
        st.error(L(lang, "Failed to load dataset.", "Không thể tải dữ liệu."))
        if settings.show_debug_errors and data_result["error"]:
            st.error(data_result["error"])
        st.stop()
    if df is None:
        st.info(
            L(
                lang,
                "Upload a PhiUSIIL CSV in the sidebar, or place the file at `data/PhiUSIIL_Phishing_URL_Dataset.csv`.",
                "Hãy upload CSV PhiUSIIL ở sidebar, hoặc đặt file tại `data/PhiUSIIL_Phishing_URL_Dataset.csv`.",
            )
        )
        st.stop()
    try:
        validate_schema(df)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    with render_section(f"1. {L(lang, 'Introduction', 'Giới thiệu')}", "intro"):
        render_intro_section(lang, source)

    with render_section(f"2. {L(lang, 'Dataset Overview', 'Tổng quan dữ liệu')}", "data_overview"):
        render_dataset_overview_section(df)

    training_drift = False
    params: TrainingUiParams = {
        "n_bins": 5,
        "bin_strategy": "quantile",
        "tld_top_n": 50,
        "test_size": 0.2,
        "max_depth": 12,
        "min_samples_split": 100,
        "row_limit": 0,
    }
    train_btn = False
    with render_section(f"3. {L(lang, 'Training Controls', 'Điều khiển huấn luyện')}", "training"):
        params, training_drift, train_btn = render_training_controls_section(df, lang)

    if train_btn:
        try:
            with st.spinner(L(lang, "Fitting preprocessing + training ID3…", "Đang fit preprocessing và huấn luyện ID3…")):
                artifacts = run_training(
                    df,
                    n_bins=params["n_bins"],
                    bin_strategy=params["bin_strategy"],
                    tld_top_n=params["tld_top_n"],
                    test_size=params["test_size"],
                    max_depth=params["max_depth"],
                    min_samples_split=params["min_samples_split"],
                    row_limit=params["row_limit"],
                )
            session_updates: dict[str, object] = dict(artifacts)
            session_updates["training_config"] = current_training_config_from_ui(
                df,
                n_bins=params["n_bins"],
                bin_strategy=params["bin_strategy"],
                categorical_top_n=params["tld_top_n"],
                test_size=params["test_size"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                row_limit=params["row_limit"],
            )
            for k, v in session_updates.items():
                st.session_state[k] = v
            st.success(L(lang, "Training finished.", "Huấn luyện xong."))
        except ValueError as e:
            st.error(L(lang, f"Target normalization failed: {e}", f"Lỗi chuẩn hóa `label`: {e}"))
            st.stop()
        except Exception as e:
            st.error(L(lang, f"Training failed: {e}", f"Huấn luyện thất bại: {e}"))
            if settings.show_debug_errors:
                st.exception(e)
            st.stop()

    with render_section(f"4. {L(lang, 'Preprocessing Summary', 'Tóm tắt tiền xử lý')}", "preprocess_summary"):
        render_preprocess_summary_section(lang)

    with render_section(f"5. {L(lang, 'Data Processing Pipeline', 'Quy trình xử lý dữ liệu')}", "pipeline"):
        render_pipeline_section(df, lang, params)

    with render_section(f"6. {L(lang, 'Evaluation', 'Đánh giá')}", "evaluation"):
        render_evaluation_section(lang, training_drift)

    with render_section(f"7. {L(lang, 'Prediction & rule explanation', 'Dự đoán & giải thích luật')}", "prediction"):
        render_prediction_section(lang, training_drift)

    st.divider()
    st.caption(L(lang, f"Project root: `{project_root()}`", f"Thư mục project: `{project_root()}`"))


if __name__ == "__main__":
    main()
