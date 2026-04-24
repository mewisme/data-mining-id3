from __future__ import annotations

import pandas as pd
import streamlit as st
from src.preprocessing import PreprocessConfig, PreprocessingPipeline
from src.ui.sections.training_controls import TrainingUiParams
from src.ui.charts import (
    plot_bar,
    plot_before_after_category,
    plot_bin_counts,
    plot_feature_space,
    plot_numeric_before_after,
    plot_split_distribution,
)
from src.ui.common import L
from src.utils import TARGET_COL


def _count_missing_like(s: pd.Series) -> int:
    as_str = s.astype(str).str.strip().str.lower()
    return int(s.isna().sum() + ((~s.isna()) & (as_str == "nan")).sum())


def _render_raw_data_section(df: pd.DataFrame, lang: str) -> None:
    st.subheader(L(lang, "5.1. Raw Data Overview", "5.1. Tổng quan dữ liệu thô"))
    c1, c2, c3 = st.columns(3)
    c1.metric(L(lang, "Rows", "Số dòng"), f"{len(df):,}")
    c2.metric(L(lang, "Columns", "Số cột"), len(df.columns))
    c3.metric(L(lang, "Target", "Nhãn đích"), TARGET_COL if TARGET_COL in df.columns else L(lang, "missing", "thiếu"))
    if TARGET_COL in df.columns:
        cls = df[TARGET_COL].value_counts(dropna=False).rename_axis("label").reset_index(name="count")
        with st.expander(L(lang, "Chart: Class distribution (raw target)", "Biểu đồ: Phân phối lớp (nhãn thô)"), expanded=False):
            st.plotly_chart(
                plot_bar(
                    cls,
                    x="label",
                    y="count",
                    title=L(lang, "Class distribution (raw target)", "Phân phối lớp (nhãn thô)"),
                ),
                width="stretch",
            )


def render_pipeline_section(df: pd.DataFrame, lang: str, params: TrainingUiParams) -> None:
    _render_raw_data_section(df, lang)

    st.subheader(L(lang, "5.2. Feature Selection / Dropped Columns", "5.2. Chọn feature / cột bị loại"))
    demo_pipe = PreprocessingPipeline(
        config=PreprocessConfig(
            drop_high_card_text=True,
            n_bins=params["n_bins"],
            bin_strategy=params["bin_strategy"],
            tld_top_n=params["tld_top_n"],
        )
    )
    decisions_df = demo_pipe.feature_decisions([str(c) for c in df.columns]).copy()
    if lang == "Tiếng Việt":
        status_map = {
            "used": "được dùng",
            "dropped": "bị loại",
        }
        reason_map = {
            "kept as model feature": "giữ lại làm đặc trưng cho mô hình",
            "target column": "cột nhãn đích",
            "identifier column": "cột định danh",
            "high-cardinality text": "cột văn bản có độ phân biệt cao",
        }
        decisions_df["status"] = decisions_df["status"].map(lambda v: status_map.get(str(v), str(v)))
        decisions_df["reason"] = decisions_df["reason"].map(lambda v: reason_map.get(str(v), str(v)))
        decisions_df = decisions_df.rename(
            columns={
                "column": "cột",
                "status": "trạng thái",
                "reason": "lý do",
            }
        )
    st.dataframe(decisions_df, width="stretch")

    st.subheader(L(lang, "5.3. Train/Test Split Visualization", "5.3. Trực quan train/test split"))
    if "split_info" in st.session_state and "train_df" in st.session_state and "test_df" in st.session_state:
        split_info = st.session_state["split_info"]
        train_df: pd.DataFrame = st.session_state["train_df"]
        test_df: pd.DataFrame = st.session_state["test_df"]
        c_left, c_right = st.columns(2)
        with c_left:
            size_df = pd.DataFrame({"split": ["train", "test"], "rows": [split_info["rows_train"], split_info["rows_test"]]})
            with st.expander(L(lang, "Chart: Train vs test size", "Biểu đồ: Kích thước train/test"), expanded=False):
                st.plotly_chart(
                    plot_bar(
                        size_df,
                        x="split",
                        y="rows",
                        title=L(lang, "Train vs test size", "Kích thước train so với test"),
                    ),
                    width="stretch",
                )
        with c_right:
            with st.expander(L(lang, "Chart: Split distribution", "Biểu đồ: Phân phối train/test"), expanded=False):
                st.plotly_chart(
                    plot_split_distribution(
                        st.session_state["work_df"],
                        train_df,
                        test_df,
                        title=L(lang, "Class balance: full vs train vs test", "Cân bằng lớp: full so với train và test"),
                    ),
                    width="stretch",
                )

    st.subheader(L(lang, "5.4. Numeric Feature Binning / Discretization", "5.4. Rời rạc hóa cột số"))
    if "pipe" in st.session_state and "train_df" in st.session_state:
        pipe: PreprocessingPipeline = st.session_state["pipe"]
        train_df = st.session_state["train_df"]
        bin_details = pipe.numeric_binning_details()
        if bin_details:
            full_stage = pipe.transform_debug_stages(train_df)
            summary_rows: list[dict[str, object]] = []
            auto_warnings: list[str] = []
            for feature in sorted(bin_details.keys()):
                detail = bin_details[feature]
                transformed_col = full_stage["transformed"][feature].astype(str)
                dominant_bin = transformed_col.value_counts().idxmax()
                dominant_share = float((transformed_col == dominant_bin).mean())
                missing_before = _count_missing_like(train_df[feature])
                missing_after = int(full_stage["missing_handled"][feature].isna().sum())
                summary_rows.append(
                    {
                        L(lang, "Feature", "Đặc trưng"): feature,
                        L(lang, "Requested bins", "Số bin yêu cầu"): int(detail["requested_bins"]),
                        L(lang, "Effective bins", "Số bin thực tế"): int(detail["effective_bins"]),
                        L(lang, "Missing (before)", "Thiếu (trước xử lý)"): missing_before,
                        L(lang, "Missing (after)", "Thiếu (sau xử lý)"): missing_after,
                        L(lang, "Dominant bin", "Bin chiếm ưu thế"): str(dominant_bin),
                        L(lang, "Dominant share", "Tỷ lệ bin ưu thế"): f"{dominant_share:.1%}",
                    }
                )
                if int(detail["effective_bins"]) < int(detail["requested_bins"]):
                    auto_warnings.append(
                        L(
                            lang,
                            f"`{feature}` uses fewer effective bins ({detail['effective_bins']}) than requested ({detail['requested_bins']}).",
                            f"`{feature}` có số bin thực tế ({detail['effective_bins']}) thấp hơn số bin yêu cầu ({detail['requested_bins']}).",
                        )
                    )
                if dominant_share >= 0.8:
                    auto_warnings.append(
                        L(
                            lang,
                            f"`{feature}` is highly concentrated: {dominant_share:.1%} samples fall into one bin ({dominant_bin}).",
                            f"`{feature}` bị dồn mạnh: {dominant_share:.1%} mẫu rơi vào một bin ({dominant_bin}).",
                        )
                    )

            st.write(L(lang, "Numeric binning overview", "Tổng quan rời rạc hóa cột số"))
            st.dataframe(pd.DataFrame(summary_rows), width="stretch")

            if auto_warnings:
                with st.expander(L(lang, "Auto warnings", "Cảnh báo tự động"), expanded=False):
                    for warn in auto_warnings:
                        st.warning(warn)

            col = st.selectbox(
                L(lang, "Pick a numeric feature for detailed charts", "Chọn cột số để xem chi tiết biểu đồ"),
                options=sorted(bin_details.keys()),
            )
            c1, c2 = st.columns(2)
            with c1:
                with st.expander(L(lang, "Chart: Before/after numeric binning", "Biểu đồ: Trước/sau rời rạc hóa"), expanded=False):
                    st.plotly_chart(
                        plot_numeric_before_after(
                            train_df,
                            full_stage,
                            col,
                            bin_details[col]["edges"],
                            title=L(lang, f"{col}: raw/imputed distribution with bin edges", f"{col}: phân phối thô/đã điền với các ngưỡng bin"),
                        ),
                        width="stretch",
                    )
            with c2:
                with st.expander(L(lang, "Chart: Bin frequencies", "Biểu đồ: Tần suất các bin"), expanded=False):
                    st.plotly_chart(
                        plot_bin_counts(
                            full_stage["transformed"][col],
                            L(lang, f"{col}: bin frequencies", f"{col}: tần suất các bin"),
                        ),
                        width="stretch",
                    )

    st.subheader(L(lang, "5.5. Categorical Transformation", "5.5. Biến đổi cột phân loại"))
    if "pipe" in st.session_state and "train_df" in st.session_state:
        pipe = st.session_state["pipe"]
        train_df = st.session_state["train_df"]
        if pipe.categorical_columns:
            cat_col = st.selectbox(L(lang, "Categorical column example", "Cột phân loại ví dụ"), options=pipe.categorical_columns)
            full_cat_stage = pipe.transform_debug_stages(train_df)
            with st.expander(L(lang, "Chart: Categorical before/after transform", "Biểu đồ: Trước/sau biến đổi cột phân loại"), expanded=False):
                st.plotly_chart(
                    plot_before_after_category(
                        train_df[cat_col],
                        full_cat_stage["transformed"][cat_col],
                        title=L(lang, "Category frequency before vs after", "Tần suất category trước và sau biến đổi"),
                    ),
                    width="stretch",
                )

    st.subheader(L(lang, "5.6. Final Model Input Preview", "5.6. Xem trước dữ liệu vào mô hình"))
    if "X_train" in st.session_state and "X_test" in st.session_state:
        X_train_df: pd.DataFrame = st.session_state["X_train"]
        feat_pick = st.selectbox(
            L(lang, "Transformed feature distribution", "Phân phối feature sau biến đổi"),
            options=list(X_train_df.columns),
            key="transformed_feature_pick",
        )
        freq_df = X_train_df[feat_pick].astype(str).value_counts().rename_axis("token").reset_index(name="count")
        with st.expander(L(lang, "Chart: X_train token frequency", "Biểu đồ: Tần suất token của X_train"), expanded=False):
            st.plotly_chart(
                plot_bar(
                    freq_df,
                    x="token",
                    y="count",
                    title=L(lang, f"X_train token frequency: {feat_pick}", f"Tần suất token của X_train: {feat_pick}"),
                ),
                width="stretch",
            )

    st.subheader(L(lang, "5.7. Feature Space Explorer (2D / 3D)", "5.7. Khám phá không gian đặc trưng (2D / 3D)"))
    if "work_df" in st.session_state and "pipe" in st.session_state:
        work_df: pd.DataFrame = st.session_state["work_df"]
        pipe: PreprocessingPipeline = st.session_state["pipe"]
        mode_space = st.radio(
            L(lang, "Feature space mode", "Chế độ không gian đặc trưng"),
            ["raw_numeric", "transformed"],
            format_func=lambda k: L(lang, "Raw numeric", "Số thô") if k == "raw_numeric" else L(lang, "Transformed", "Đã biến đổi"),
            horizontal=True,
        )
        if mode_space == "raw_numeric":
            plot_df = work_df.copy()
            numeric_opts = [c for c in pipe.numeric_columns if c in plot_df.columns]
            for c in numeric_opts:
                plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")
        else:
            transformed_full = pipe.transform_X(work_df)
            plot_df = transformed_full.copy()
            numeric_opts = []
            for c in transformed_full.columns:
                plot_df[c] = pd.factorize(transformed_full[c].astype(str))[0]
                numeric_opts.append(c)
        if len(numeric_opts) >= 2:
            c1, c2, c3 = st.columns(3)
            x_feat = c1.selectbox(L(lang, "X feature", "Feature trục X"), options=numeric_opts, index=0)
            y_feat = c2.selectbox(L(lang, "Y feature", "Feature trục Y"), options=numeric_opts, index=1)
            use_3d = c3.checkbox(L(lang, "Enable 3D", "Bật 3D"), value=False)
            z_feat = (
                st.selectbox(
                    L(lang, "Z feature (3D)", "Feature trục Z (3D)"),
                    options=numeric_opts,
                    index=min(2, len(numeric_opts) - 1),
                )
                if use_3d
                else None
            )
            plot_df["class_label"] = work_df[TARGET_COL].astype(str).values if TARGET_COL in work_df.columns else "unknown"
            with st.expander(L(lang, "Chart: Feature space explorer", "Biểu đồ: Không gian đặc trưng"), expanded=False):
                st.plotly_chart(
                    plot_feature_space(
                        plot_df,
                        x=x_feat,
                        y=y_feat,
                        z=z_feat,
                        color="class_label",
                        title_2d=L(lang, "Feature space explorer (2D)", "Khám phá không gian đặc trưng (2D)"),
                        title_3d=L(lang, "Feature space explorer (3D)", "Khám phá không gian đặc trưng (3D)"),
                    ),
                    width="stretch",
                )
