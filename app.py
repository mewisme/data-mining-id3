"""
Streamlit UI: PhiUSIIL phishing URL classification with custom ID3.
Run: streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from src.data_loader import load_csv_from_path, load_csv_from_upload, resolve_default_path, validate_schema
from src.evaluation import confusion_matrix_df, evaluate, report_string
from src.id3 import ID3Classifier
from src.preprocessing import PreprocessConfig, PreprocessingPipeline, normalize_target, preprocessing_summary
from src.predictor import format_prediction, predict_manual, predict_test_row
from src.utils import FEATURE_COLUMNS, ID_COL_DROP, MANUAL_PREDICTION_FEATURES, TARGET_COL, label_to_display, project_root


def tr(vi: str, en: str, lang: str) -> str:
    """Simple bilingual helper; VI keeps original English in parentheses."""
    if lang == "Tiếng Việt":
        return f"{vi} ({en})"
    return en


def _format_path_rule(path: list[tuple[str, object, str]]) -> tuple[str, str]:
    conditions: list[str] = []
    predicted_class = "unknown"
    for feat, val, _ in path:
        if feat == "class":
            predicted_class = str(val)
        else:
            conditions.append(f"{feat} = {val}")
    if conditions:
        rule = "IF " + " AND ".join(conditions) + f" THEN class = {predicted_class}"
        natural = (
            "This sample followed the branch where "
            + ", ".join(conditions)
            + f", so the model classified it as {predicted_class}."
        )
    else:
        rule = f"THEN class = {predicted_class}"
        natural = f"The tree reached a direct leaf, so the model classified it as {predicted_class}."
    return rule, natural


def _safe_bin_ranges(pipe_obj: object) -> dict[str, list[str]]:
    """Get bin ranges from new or old PreprocessingPipeline objects."""
    # New versions
    if hasattr(pipe_obj, "bin_ranges") and callable(getattr(pipe_obj, "bin_ranges")):
        try:
            return getattr(pipe_obj, "bin_ranges")()
        except Exception:
            return {}

    # Backward compatibility for older session_state objects
    out: dict[str, list[str]] = {}
    numeric_cols = getattr(pipe_obj, "numeric_columns", [])
    discretizers = getattr(pipe_obj, "discretizers", {})
    for col in numeric_cols:
        disc = discretizers.get(col) if isinstance(discretizers, dict) else None
        if disc is None or not hasattr(disc, "bin_edges_"):
            continue
        edges = disc.bin_edges_[0]
        labels: list[str] = []
        for i in range(len(edges) - 1):
            left = float(edges[i])
            right = float(edges[i + 1])
            if i < len(edges) - 2:
                labels.append(f"bin_{i}: [{left:.6g}, {right:.6g})")
            else:
                labels.append(f"bin_{i}: [{left:.6g}, {right:.6g}]")
        out[col] = labels
    return out


def _load_dataframe(uploaded) -> tuple[pd.DataFrame | None, str]:
    """Load default local CSV first; fallback to uploaded CSV."""
    path = resolve_default_path()
    try:
        if path.is_file():
            return load_csv_from_path(path), "default"
        if uploaded is not None:
            return load_csv_from_upload(uploaded), "upload"
        return None, "none"
    except Exception as e:
        st.error(str(e))
        return None, "error"


def _dataset_overview(df: pd.DataFrame, lang: str) -> None:
    st.header(f"2. {tr('Tổng quan dữ liệu', 'Dataset Overview', lang)}")
    n = len(df)
    st.metric(tr("Tổng số dòng", "Total rows", lang), f"{n:,}")
    st.metric(tr("Tổng số cột", "Total columns", lang), len(df.columns))
    feat_count = len([c for c in FEATURE_COLUMNS if c in df.columns])
    st.metric(tr("Số cột đặc trưng theo schema", "Schema feature columns present", lang), feat_count)
    if TARGET_COL in df.columns:
        st.write(f"**{tr('Phân bố biến mục tiêu', 'Target distribution', lang)} (`label`)**")
        vc = df[TARGET_COL].value_counts().sort_index()
        st.bar_chart(vc)
        st.write(vc.to_dict())
    st.write(f"**{tr('Xem trước dữ liệu', 'Preview', lang)}**")
    st.dataframe(df.head(15), width="stretch")
    st.write(f"**{tr('Giá trị thiếu (các cột nhiều nhất)', 'Missing values (top columns)', lang)}**")
    miss = df.isna().sum().sort_values(ascending=False)
    st.dataframe(miss[miss > 0].head(25), width="stretch")
    st.write(f"**{tr('Kiểu dữ liệu cột', 'Column dtypes', lang)}**")
    st.dataframe(df.dtypes.astype(str).to_frame("dtype"), width="stretch")
    dtype_summary = df.dtypes.astype(str).value_counts().rename_axis("dtype").reset_index(name="count")
    st.write(f"**{tr('Tóm tắt kiểu đặc trưng', 'Feature type summary', lang)}**")
    st.dataframe(dtype_summary, width="stretch")


def main() -> None:
    st.set_page_config(page_title="ID3 Phishing URL Classifier", layout="wide")
    with st.sidebar:
        lang = st.selectbox("Language / Ngôn ngữ", ["English", "Tiếng Việt"], index=0)

    st.title(tr("Phân loại URL phishing bằng ID3 (PhiUSIIL)", "Phishing URL classification with ID3 (PhiUSIIL)", lang))
    st.header(f"1. {tr('Giới thiệu', 'Introduction', lang)}")
    st.markdown(
        f"""
**{tr('Dự án', 'Project', lang)}:** {tr('Phân loại nhị phân URL / đặc trưng trang web thành', 'Binary classification of URLs / page features into', lang)} **phishing** {tr('và', 'vs', lang)} **legitimate**
using the **PhiUSIIL Phishing URL Dataset** and a **custom ID3** decision tree
({tr('entropy', 'entropy', lang)}, {tr('information gain', 'information gain', lang)}, {tr('tách đệ quy', 'recursive splits', lang)} — {tr('không dùng cây', 'not using', lang)} `sklearn`).

**{tr('URL phishing', 'Phishing URLs', lang)}** {tr('giả mạo website tin cậy để đánh cắp thông tin; phân loại tự động từ tín hiệu URL/trang hỗ trợ phòng thủ.', 'mimic trusted sites to steal credentials; automated classification from URL and page signals supports defensive tooling.', lang)}

**ID3** {tr('xây cây bằng cách chọn phép tách tối đa', 'builds a tree by choosing splits that maximize', lang)} **{tr('lợi ích thông tin (information gain)', 'information gain', lang)}** {tr('từ', 'from', lang)} **{tr('độ hỗn loạn (entropy)', 'entropy', lang)}**;
{tr('ID3 cần thuộc tính rời rạc (discrete), nên cột số được rời rạc hóa (binning) chỉ trên tập train.', 'it expects discrete attributes, so numeric fields are binned on the training set only.', lang)}
        """  # noqa: E501
    )
    st.info(
        tr(
            "Mô hình: cây quyết định ID3 tự cài đặt. Tiêu chí tách: entropy và information gain. Không dùng sklearn DecisionTreeClassifier.",
            "Model: Custom ID3 decision tree. Splitting criterion: entropy and information gain. This is not sklearn DecisionTreeClassifier.",
            lang,
        )
    )

    with st.sidebar:
        st.header(tr("Nguồn dữ liệu", "Data source", lang))
        uploaded = st.file_uploader("PhiUSIIL CSV", type=["csv"])
        default_path = resolve_default_path()
        if default_path.is_file():
            st.caption(
                tr(
                    f"Đã phát hiện file mặc định: {default_path}. Ứng dụng sẽ ưu tiên dùng file này.",
                    f"Detected default dataset: {default_path}. The app will use it by default.",
                    lang,
                )
            )
        else:
            st.caption(
                tr(
                    "Không tìm thấy file mặc định. Vui lòng tải lên CSV để sử dụng.",
                    "Default dataset not found. Please upload a CSV to run.",
                    lang,
                )
            )

    df, source = _load_dataframe(uploaded)

    if df is None:
        st.info(
            tr(
                "Vui lòng tải lên file CSV PhiUSIIL trong sidebar để bắt đầu.",
                "Please upload a PhiUSIIL CSV in the sidebar to start.",
                lang,
            )
        )
        st.stop()
    if source == "default":
        st.success(
            tr(
                f"Đang dùng dữ liệu mặc định từ: {resolve_default_path()}",
                f"Using default dataset from: {resolve_default_path()}",
                lang,
            )
        )
    elif source == "upload":
        st.success(tr("Đang dùng dữ liệu từ file upload.", "Using dataset from uploaded file.", lang))

    try:
        validate_schema(df)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # --- Dataset overview ---
    _dataset_overview(df, lang)

    # --- Preprocessing controls ---
    st.header(f"3. {tr('Tiền xử lý', 'Preprocessing', lang)}")
    drop_high = st.checkbox(
        tr("Loại cột text có cardinality cao", "Drop high-cardinality text", lang) + ": `URL`, `Domain`, `Title` (" + tr("khuyến nghị", "recommended", lang) + ")",
        value=True,
        help=tr(
            "Các cột text thô gần như duy nhất mỗi dòng; ID3 sẽ rất lớn và chậm. TLD được giữ với gom nhóm top-N.",
            "Raw text fields have near-unique values; ID3 becomes huge and slow. TLD is kept with top-N bucketing.",
            lang,
        ),
    )
    n_bins = st.slider(tr("Số lượng bins (rời rạc hóa số)", "Number of bins (numeric discretization)", lang), 3, 15, 5)
    bin_strategy = st.selectbox(tr("Chiến lược chia bins", "Bin strategy", lang), ["quantile", "uniform"], index=0)
    tld_top_n = st.slider(tr("TLD: giữ top-N (còn lại → OTHER)", "TLD: keep top-N categories (rest → OTHER)", lang), 10, 200, 50)
    if bin_strategy == "quantile":
        st.caption(
            tr(
                "Bạn chọn quantile: mỗi bin sẽ có số lượng mẫu gần tương đương nhau (không nhất thiết cùng độ rộng giá trị).",
                "Selected strategy = quantile: each bin contains a similar number of samples (value ranges may have different widths).",
                lang,
            )
        )
    else:
        st.caption(
            tr(
                "Bạn chọn uniform: các bin có cùng độ rộng khoảng giá trị (số lượng mẫu trong mỗi bin có thể rất khác nhau).",
                "Selected strategy = uniform: bins have equal value-range width (sample counts per bin may vary a lot).",
                lang,
            )
        )

    st.markdown(
        f"- **{tr('Cột định danh bị loại', 'Dropped identifier', lang)}:** `{ID_COL_DROP}`\n"
        f"- **{tr('Biến mục tiêu', 'Target', lang)}:** `{TARGET_COL}` (0 = legitimate, 1 = phishing)\n"
        f"- **{tr('Cột text cardinality cao bị loại mặc định', 'Dropped high-cardinality text columns (default)', lang)}:** `URL`, `Domain`, `Title`\n"
        f"- **{tr('Cột hạng mục giữ lại', 'Kept categorical column', lang)}:** `TLD` (top-{tld_top_n} + OTHER)\n"
        f"- **{tr('Rời rạc hóa số', 'Numeric discretization', lang)}:** {bin_strategy}, bins={n_bins}, {tr('fit trên train và tái sử dụng cho test/predict', 'fit on train and reused for test/predict', lang)}"
    )

    # --- Training ---
    st.header(f"4. {tr('Huấn luyện mô hình', 'Model training', lang)}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        test_size = st.slider(tr("Tỉ lệ tập kiểm tra", "Test size", lang), 0.1, 0.4, 0.2, 0.05)
    with col2:
        max_depth = st.number_input(tr("Độ sâu tối đa", "Max depth", lang), min_value=1, max_value=50, value=12)
    with col3:
        min_samples_split = st.number_input(tr("Số mẫu tối thiểu để tách", "Min samples split", lang), min_value=2, max_value=5000, value=100)
    with col4:
        row_limit = st.number_input(tr("Giới hạn số dòng (0 = toàn bộ)", "Row limit (0 = all rows)", lang), min_value=0, max_value=500_000, value=8000, step=500)

    train_btn = st.button(tr("Huấn luyện mô hình ID3", "Train ID3 model", lang), type="primary")

    if train_btn:
        work = df.copy()
        try:
            y_norm, _ = normalize_target(work[TARGET_COL])
            work[TARGET_COL] = y_norm
        except Exception as e:
            st.error(f"Target normalization failed: {e}")
            st.stop()

        if row_limit and row_limit > 0:
            work = work.sample(n=min(row_limit, len(work)), random_state=42).reset_index(drop=True)
            st.info(
                tr(
                    f"Huấn luyện trên **{len(work):,}** dòng (đã áp dụng giới hạn). Đây là tập demo, không phải toàn bộ dữ liệu.",
                    f"Training on **{len(work):,}** rows (row limit applied). This is a demo subset, not full data.",
                    lang,
                )
            )

        try:
            train_df, test_df = train_test_split(
                work,
                test_size=test_size,
                random_state=42,
                stratify=work[TARGET_COL],
            )
        except ValueError:
            train_df, test_df = train_test_split(work, test_size=test_size, random_state=42)

        cfg = PreprocessConfig(
            drop_high_card_text=drop_high,
            n_bins=n_bins,
            bin_strategy=bin_strategy,  # type: ignore[arg-type]
            tld_top_n=tld_top_n,
        )
        pipe = PreprocessingPipeline(config=cfg)
        with st.spinner("Fitting preprocessing & training ID3..."):
            pipe.fit(train_df)
            X_train = pipe.transform_X(train_df)
            y_train = train_df[TARGET_COL].astype(int).values
            X_test = pipe.transform_X(test_df)
            y_test = test_df[TARGET_COL].astype(int).values

            model = ID3Classifier(max_depth=int(max_depth), min_samples_split=int(min_samples_split))
            model.fit(X_train, pd.Series(y_train))

        st.session_state["pipe"] = pipe
        st.session_state["model"] = model
        st.session_state["test_df"] = test_df
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["preprocess_info"] = preprocessing_summary(cfg, pipe.feature_columns)
        st.session_state["sampling_info"] = {
            "row_limit_requested": int(row_limit),
            "row_sampling_enabled": bool(row_limit and row_limit > 0),
            "rows_used_for_training_pipeline": int(len(work)),
            "rows_original": int(len(df)),
            "binning_fit_train_only": True,
        }
        st.success(tr("Huấn luyện hoàn tất.", "Training finished.", lang))

    st.subheader(tr("Tóm tắt tiền xử lý (lần train gần nhất)", "Preprocessing summary (last train)", lang))
    if "preprocess_info" in st.session_state:
        info = st.session_state["preprocess_info"]
        sample_info = st.session_state.get("sampling_info", {})
        pipe_for_bins = st.session_state.get("pipe")
        st.markdown(
            f"- **{tr('Cột định danh bị loại', 'Dropped identifier', lang)}:** `{info['dropped_identifier']}`\n"
            f"- **{tr('Biến mục tiêu', 'Target', lang)}:** `{info['target']}`\n"
            f"- **{tr('Cột text cardinality cao bị loại mặc định', 'Dropped high-cardinality text columns (default)', lang)}:** "
            + (", ".join(f"`{c}`" for c in info["dropped_high_card_text_default"]) or tr("Không", "None", lang))
            + "\n"
            f"- **{tr('Cột hạng mục giữ lại', 'Kept categorical column(s)', lang)}:** "
            + (", ".join(f"`{c}`" for c in info["categorical_kept"]) or tr("Không", "None", lang))
            + "\n"
            f"- **{tr('Chiến lược rời rạc hóa số', 'Numeric discretization strategy', lang)}:** `{info['bin_strategy']}`, bins=`{info['n_bins']}`\n"
            f"- **{tr('Binning fit trên train-only', 'Binning fit on train-only', lang)}:** `{sample_info.get('binning_fit_train_only', True)}`\n"
            f"- **{tr('Giới hạn dòng/sampling', 'Row limit/sampling', lang)}:** `{sample_info.get('row_sampling_enabled', False)}` "
            f"({tr('dùng', 'used', lang)} {sample_info.get('rows_used_for_training_pipeline', 'N/A')} / {sample_info.get('rows_original', 'N/A')} {tr('dòng', 'rows', lang)})"
        )
        if pipe_for_bins is not None:
            with st.expander(tr("Giải thích khoảng chia bins (numeric split ranges)", "Explain bin split ranges (numeric features)", lang)):
                st.caption(
                    tr(
                        "Các khoảng này được fit trên tập train. Ví dụ: nếu một mẫu có giá trị rơi vào khoảng của bin_2 thì trong cây ID3 sẽ dùng nhánh 'bin_2'.",
                        "These ranges are fit on the train set. Example: if a sample value falls in bin_2 range, ID3 uses branch 'bin_2'.",
                        lang,
                    )
                )
                bin_map = _safe_bin_ranges(pipe_for_bins)
                if not bin_map:
                    st.write(tr("Không có cột số để hiển thị bins.", "No numeric bin ranges available.", lang))
                else:
                    chosen = st.selectbox(
                        tr("Chọn cột số để xem bins", "Select numeric feature for bins", lang),
                        options=sorted(bin_map.keys()),
                    )
                    st.write(f"**{chosen}**")
                    for line in bin_map[chosen]:
                        st.code(line)
    else:
        st.caption(tr("Huấn luyện mô hình để xem chi tiết tiền xử lý.", "Train the model to see preprocessing details.", lang))

    # --- Evaluation ---
    st.header(f"5. {tr('Đánh giá', 'Evaluation', lang)}")
    if "model" not in st.session_state:
        st.caption(tr("Hãy huấn luyện mô hình trước.", "Train the model first.", lang))
    else:
        model: ID3Classifier = st.session_state["model"]
        X_test: pd.DataFrame = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        y_pred = model.predict(X_test)
        metrics = evaluate(y_test, y_pred)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(tr("Độ chính xác", "Accuracy", lang), f"{metrics['accuracy']:.4f}")
        m2.metric(tr("Độ chuẩn xác (phishing)", "Precision (phishing)", lang), f"{metrics['precision']:.4f}")
        m3.metric(tr("Độ bao phủ (phishing)", "Recall (phishing)", lang), f"{metrics['recall']:.4f}")
        m4.metric(tr("Điểm F1", "F1-score", lang), f"{metrics['f1']:.4f}")
        st.caption(tr("Quy ước nhãn: 0 = legitimate, 1 = phishing.", "Label mapping: 0 = legitimate, 1 = phishing.", lang))
        st.write(f"**{tr('Ma trận nhầm lẫn', 'Confusion matrix', lang)}**")
        st.dataframe(confusion_matrix_df(y_test, y_pred), width="stretch")
        st.text(tr("Báo cáo phân loại", "Classification report", lang))
        st.text(report_string(y_test, y_pred))

    # --- Prediction & rule explanation ---
    st.header(f"6. {tr('Dự đoán và giải thích luật', 'Prediction & rule explanation', lang)}")
    if "model" not in st.session_state or "test_df" not in st.session_state:
        st.caption(tr("Huấn luyện mô hình để bật chức năng dự đoán.", "Train the model to enable prediction.", lang))
    else:
        pipe = st.session_state["pipe"]
        model = st.session_state["model"]
        test_df: pd.DataFrame = st.session_state["test_df"]

        mode = st.radio("Mode", ["Choose test row (default)", "Manual features (subset)"], horizontal=True)

        if mode.startswith("Choose"):
            idx = st.number_input("Test set row index", min_value=0, max_value=len(test_df) - 1, value=0)
            row = test_df.iloc[int(idx)]
            st.write("**Raw row (excerpt)**")
            show_cols = [c for c in pipe.feature_columns if c in row.index][:20]
            st.dataframe(row[show_cols].to_frame().T, width="stretch")
            if st.button("Predict this row"):
                pred, path = predict_test_row(model, pipe, row)
                if TARGET_COL in row.index:
                    st.write(f"**True label:** {int(row[TARGET_COL])} ({label_to_display(int(row[TARGET_COL]))})")
                st.write(f"**Prediction:** {format_prediction(pred)}")
                st.write("**Decision path / explanation**")
                for step in path:
                    st.write(f"- {step[0]} = {step[1]} → {step[2]}")
                rule_text, natural_text = _format_path_rule(path)
                st.write("**Readable IF-THEN rule**")
                st.code(rule_text)
                st.write(f"**Natural-language explanation:** {natural_text}")
                st.write("**Sample IF-THEN rules (tree prefix)**")
                for r in model.rules_to_text(max_rules=16):
                    st.code(r)

        else:
            st.caption("Only a subset of features is required; other fields use training defaults.")
            updates: dict = {}
            cols = st.columns(2)
            manual_fields = [f for f in MANUAL_PREDICTION_FEATURES if f in pipe.feature_columns]
            assert pipe.default_raw_row is not None
            for i, fname in enumerate(manual_fields):
                with cols[i % 2]:
                    if fname == "TLD":
                        default_tld = str(pipe.default_raw_row.get(fname, "com"))
                        updates[fname] = st.text_input("TLD (e.g. com, de)", value=default_tld)
                    else:
                        dv = float(pipe.default_raw_row[fname])
                        updates[fname] = st.number_input(fname, value=dv, format="%.6f")
            if st.button("Predict manual"):
                # TLD as string; numeric as float
                manual_updates = {}
                for k, v in updates.items():
                    if k == "TLD":
                        manual_updates[k] = str(v).strip() or "com"
                    else:
                        manual_updates[k] = float(v)
                pred, path = predict_manual(model, pipe, manual_updates)
                st.write(f"**Prediction:** {format_prediction(pred)}")
                for step in path:
                    st.write(f"- {step[0]} = {step[1]} → {step[2]}")
                rule_text, natural_text = _format_path_rule(path)
                st.write("**Readable IF-THEN rule**")
                st.code(rule_text)
                st.write(f"**Natural-language explanation:** {natural_text}")
                for r in model.rules_to_text(max_rules=8):
                    st.code(r)

    st.divider()
    st.caption(f"Project root: `{project_root()}`")


if __name__ == "__main__":
    main()
