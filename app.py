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
    """Vietnamese prose in `vi`; English in `en`. Technical terms often stay English inside `vi`."""
    if lang == "Tiếng Việt":
        if vi.strip() == en.strip():
            return en
        return vi
    return en


def L(lang: str, en: str, vi: str) -> str:
    """Readable alias: English first in signature."""
    return tr(vi, en, lang)


def _format_path_rule(path: list[tuple[str, object, str]], lang: str) -> tuple[str, str]:
    conditions: list[str] = []
    predicted_class = "unknown"
    for feat, val, _ in path:
        if feat == "class":
            predicted_class = str(val)
        else:
            conditions.append(f"{feat} = {val}")
    if conditions:
        rule = "IF " + " AND ".join(conditions) + f" THEN class = {predicted_class}"
        cond_txt = ", ".join(conditions)
        if lang == "Tiếng Việt":
            natural = f"Mẫu đi qua các nhánh {cond_txt}, nên mô hình gán nhãn **{predicted_class}**."
        else:
            natural = f"This sample followed the branch where {cond_txt}, so the model classified it as **{predicted_class}**."
    else:
        rule = f"THEN class = {predicted_class}"
        if lang == "Tiếng Việt":
            natural = f"Cây kết thúc ngay tại lá (leaf), mô hình gán nhãn **{predicted_class}**."
        else:
            natural = f"The tree reached a direct leaf, so the model classified it as **{predicted_class}**."
    return rule, natural


def _training_config_snapshot(
    *,
    n_bins: int,
    bin_strategy: str,
    categorical_top_n: int,
    test_size: float,
    max_depth: int,
    min_samples_split: int,
    row_limit: int,
    data_row_count: int,
    data_columns_fingerprint: tuple[str, ...],
) -> dict[str, object]:
    """Hashable training UI + data fingerprint so we can detect stale metrics."""
    return {
        "n_bins": int(n_bins),
        "bin_strategy": str(bin_strategy),
        "categorical_top_n": int(categorical_top_n),
        "test_size": float(test_size),
        "max_depth": int(max_depth),
        "min_samples_split": int(min_samples_split),
        "row_limit": int(row_limit),
        "data_row_count": int(data_row_count),
        "data_columns_fingerprint": data_columns_fingerprint,
    }


def _current_training_config_from_ui(
    df: pd.DataFrame,
    *,
    n_bins: int,
    bin_strategy: str,
    categorical_top_n: int,
    test_size: float,
    max_depth: int,
    min_samples_split: int,
    row_limit: int,
) -> dict[str, object]:
    return _training_config_snapshot(
        n_bins=n_bins,
        bin_strategy=bin_strategy,
        categorical_top_n=categorical_top_n,
        test_size=test_size,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        row_limit=row_limit,
        data_row_count=len(df),
        data_columns_fingerprint=tuple(sorted(df.columns.astype(str))),
    )


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
    st.header(f"2. {L(lang, 'Dataset Overview', 'Tổng quan dữ liệu')}")
    n = len(df)
    st.metric("Total rows", f"{n:,}")
    st.metric("Total columns", len(df.columns))
    feat_count = len([c for c in FEATURE_COLUMNS if c in df.columns])
    st.metric("Schema features (present)", feat_count)
    if TARGET_COL in df.columns:
        st.write("**Target distribution** (`label`)")
        vc = df[TARGET_COL].value_counts().sort_index()
        st.bar_chart(vc)
        st.write(vc.to_dict())
    st.write("**Preview**")
    st.dataframe(df.head(15), width="stretch")
    st.write("**Missing values** (top columns)")
    miss = df.isna().sum().sort_values(ascending=False)
    st.dataframe(miss[miss > 0].head(25), width="stretch")
    st.write("**Column dtypes**")
    st.dataframe(df.dtypes.astype(str).to_frame("dtype"), width="stretch")
    dtype_summary = df.dtypes.astype(str).value_counts().rename_axis("dtype").reset_index(name="count")
    st.write("**Feature type summary**")
    st.dataframe(dtype_summary, width="stretch")


def main() -> None:
    st.set_page_config(page_title="ID3 Phishing URL Classifier", layout="wide")
    with st.sidebar:
        lang = st.selectbox("Language / Ngôn ngữ", ["English", "Tiếng Việt"], index=0)

    st.title(
        L(
            lang,
            "Phishing URL classification with ID3 (PhiUSIIL)",
            "Phân loại URL phishing bằng ID3 (PhiUSIIL)",
        )
    )
    st.header(f"1. {L(lang, 'Introduction', 'Giới thiệu')}")
    if lang == "Tiếng Việt":
        st.markdown(
            """
**Project:** phân loại nhị phân **phishing** và **legitimate** từ đặc trưng URL / trang web, trên **PhiUSIIL Phishing URL Dataset**, bằng **custom ID3 decision tree**
(**entropy**, **information gain**, **recursive splits** — không dùng `sklearn` **DecisionTreeClassifier**).

**Phishing URLs** giả mạo site tin cậy để lừa lấy thông tin; phân loại tự động hỗ trợ phòng thủ dựa trên tín hiệu URL / trang.

**ID3** chọn thuộc tính tách sao cho **information gain** (từ **entropy**) lớn nhất. Thuật toán cần đầu vào **discrete**, nên cột số được **binning** (rời rạc hóa) và **chỉ fit trên tập train**.
"""
        )
    else:
        st.markdown(
            """
**Project:** binary classification into **phishing** vs **legitimate** from URL / page features, using the **PhiUSIIL Phishing URL Dataset** and a **custom ID3 decision tree**
(**entropy**, **information gain**, **recursive splits** — not `sklearn` **DecisionTreeClassifier**).

**Phishing URLs** mimic trusted sites to steal credentials; automated classification from URL/page signals supports defensive tooling.

**ID3** picks splits that maximize **information gain** from **entropy**. It expects **discrete** inputs, so numeric columns are **binned** and **fit on the train split only**.
"""
        )
    if lang == "Tiếng Việt":
        st.info(
            "**Model:** **custom ID3 decision tree**. **Splitting criterion:** **entropy** và **information gain**. "
            "Không dùng `sklearn.tree.DecisionTreeClassifier`."
        )
    else:
        st.info(
            "**Model:** **custom ID3 decision tree**. **Splitting criterion:** **entropy** and **information gain**. "
            "Not `sklearn.tree.DecisionTreeClassifier`."
        )

    with st.sidebar:
        st.header(L(lang, "Data source", "Nguồn dữ liệu"))
        uploaded = st.file_uploader(L(lang, "PhiUSIIL CSV upload", "Tải lên CSV PhiUSIIL"), type=["csv"])
        default_path = resolve_default_path()
        if default_path.is_file():
            st.caption(
                L(
                    lang,
                    f"Default dataset found: `{default_path}`. Loaded automatically if you do not upload a file.",
                    f"Đã có file mặc định `{default_path}` — ưu tiên dùng nếu bạn không upload file khác.",
                )
            )
        else:
            st.caption(
                L(
                    lang,
                    "No default CSV in `data/`. Upload a PhiUSIIL CSV to continue.",
                    "Không thấy CSV mặc định trong `data/`. Hãy upload file PhiUSIIL.",
                )
            )

    df, source = _load_dataframe(uploaded)

    if df is None:
        st.info(
            L(
                lang,
                "Upload a PhiUSIIL CSV in the sidebar, or place the file at `data/PhiUSIIL_Phishing_URL_Dataset.csv`.",
                "Hãy upload CSV PhiUSIIL ở sidebar, hoặc đặt file tại `data/PhiUSIIL_Phishing_URL_Dataset.csv`.",
            )
        )
        st.stop()
    if source == "default":
        st.success(
            L(
                lang,
                f"Using default dataset: `{resolve_default_path()}`",
                f"Đang dùng dữ liệu mặc định: `{resolve_default_path()}`",
            )
        )
    elif source == "upload":
        st.success(L(lang, "Using uploaded CSV.", "Đang dùng file CSV đã upload."))

    try:
        validate_schema(df)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # --- Dataset overview ---
    _dataset_overview(df, lang)

    # --- Preprocessing controls ---
    st.header(f"3. {L(lang, 'Preprocessing', 'Tiền xử lý')}")
    st.info(
        L(
            lang,
            "Columns `URL`, `Domain`, and `Title` are **always dropped**: they are almost unique per row, so keeping them would make the custom ID3 tree huge, and numeric coercion would silently ruin those fields. Categorical columns (e.g. `TLD`) use **top-N** on the train split with `OTHER` for rare/unseen values.",
            "Các cột `URL`, `Domain`, `Title` **luôn bị bỏ**: gần như khác nhau mỗi dòng nên giữ lại sẽ làm cây ID3 phình to; ép kiểu số cũng làm hỏng ngữ nghĩa. Cột phân loại (vd. `TLD`) dùng **top-N** trên tập train và `OTHER` cho giá trị hiếm/chưa gặp.",
        )
    )
    n_bins = st.slider(L(lang, "Number of bins (numeric discretization)", "Số **bins** (rời rạc hóa số)"), 3, 15, 5)
    bin_strategy = st.selectbox(L(lang, "Bin strategy", "Chiến lược **bin**"), ["quantile", "uniform"], index=0)
    tld_top_n = st.slider(
        L(
            lang,
            "Top-N frequent categories per categorical column (else OTHER)",
            "**Top-N** hạng mục thường gặp / mỗi cột phân loại (còn lại **OTHER**)",
        ),
        10,
        200,
        50,
    )
    if bin_strategy == "quantile":
        st.caption(
            L(
                lang,
                "**quantile:** bins contain similar sample counts; bin width on the value axis can differ.",
                "**quantile:** các bin chứa khoảng cùng số mẫu; độ rộng trên trục giá trị có thể khác nhau.",
            )
        )
    else:
        st.caption(
            L(
                lang,
                "**uniform:** bins have equal value-range width; sample counts per bin may differ a lot.",
                "**uniform:** các bin có cùng độ rộng khoảng giá trị; số mẫu mỗi bin có thể rất khác.",
            )
        )

    st.caption(
        L(
            lang,
            "Sliders above are **for the next training run** only. Metrics and the tree reflect the **last** Train click unless you retrain.",
            "Thanh trượt phía trên chỉ áp dụng cho **lần Train tiếp theo**. Metrics và cây phản ánh **lần Train gần nhất** nếu bạn chưa huấn luyện lại.",
        )
    )
    st.markdown(
        f"- **Dropped identifier:** `{ID_COL_DROP}`\n"
        f"- **Target:** `{TARGET_COL}` (`0` = legitimate, `1` = phishing; unsupported or missing labels error out)\n"
        f"- **Always dropped (high-cardinality text):** `URL`, `Domain`, `Title`\n"
        f"- **Categorical columns:** top-{tld_top_n} per column on **train** + `OTHER`\n"
        f"- **Numeric discretization:** `{bin_strategy}`, bins=`{n_bins}`, fit on **train** only, applied to **test** / predict"
    )

    # --- Training ---
    st.header(f"4. {L(lang, 'Model training', 'Huấn luyện mô hình')}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        test_size = st.slider(L(lang, "Test split ratio", "Tỉ lệ **test split**"), 0.1, 0.4, 0.2, 0.05)
    with col2:
        max_depth = st.number_input(L(lang, "Max tree depth", "Độ sâu tối đa (**max depth**)"), min_value=1, max_value=50, value=12)
    with col3:
        min_samples_split = st.number_input(
            L(lang, "Min samples to split", "Ngưỡng **min samples split**"), min_value=2, max_value=5000, value=100
        )
    with col4:
        row_limit = st.number_input(
            L(lang, "Row limit (`0` = all rows)", "Giới hạn dòng (`0` = full data)"), min_value=0, max_value=500_000, value=8000, step=500
        )

    training_drift = False
    if "model" in st.session_state and "training_config" in st.session_state:
        current_cfg = _current_training_config_from_ui(
            df,
            n_bins=n_bins,
            bin_strategy=bin_strategy,
            categorical_top_n=tld_top_n,
            test_size=test_size,
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            row_limit=int(row_limit),
        )
        training_drift = current_cfg != st.session_state["training_config"]

    train_btn = st.button(L(lang, "Train ID3", "Huấn luyện **ID3**"), type="primary")

    if train_btn:
        work = df.copy()
        try:
            y_norm, _ = normalize_target(work[TARGET_COL])
            work[TARGET_COL] = y_norm
        except ValueError as e:
            st.error(L(lang, f"Target normalization failed: {e}", f"Lỗi chuẩn hóa `label`: {e}"))
            st.stop()

        if row_limit and row_limit > 0:
            work = work.sample(n=min(row_limit, len(work)), random_state=42).reset_index(drop=True)
            st.info(
                L(
                    lang,
                    f"Training pipeline uses **{len(work):,}** rows (**row limit**). Not the full CSV.",
                    f"Pipeline huấn luyện **{len(work):,}** dòng (**row limit**), không phải full CSV.",
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
            drop_high_card_text=True,
            n_bins=n_bins,
            bin_strategy=bin_strategy,  # type: ignore[arg-type]
            tld_top_n=tld_top_n,
        )
        pipe = PreprocessingPipeline(config=cfg)
        with st.spinner(L(lang, "Fitting preprocessing + training ID3…", "Đang fit preprocessing và huấn luyện ID3…")):
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
        st.session_state["training_config"] = _current_training_config_from_ui(
            df,
            n_bins=n_bins,
            bin_strategy=bin_strategy,
            categorical_top_n=tld_top_n,
            test_size=test_size,
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            row_limit=int(row_limit),
        )
        st.success(L(lang, "Training finished.", "Huấn luyện xong."))

    st.subheader(L(lang, "Preprocessing summary (last train)", "Tóm tắt preprocessing (train gần nhất)"))
    if "preprocess_info" in st.session_state:
        info = st.session_state["preprocess_info"]
        sample_info = st.session_state.get("sampling_info", {})
        pipe_for_bins = st.session_state.get("pipe")
        st.markdown(
            f"- **Dropped identifier:** `{info['dropped_identifier']}`\n"
            f"- **Target:** `{info['target']}`\n"
            f"- **Dropped high-cardinality text (always for this demo):** "
            + (", ".join(f"`{c}`" for c in info["dropped_high_card_text_default"]) or L(lang, "—", "—"))
            + "\n"
            f"- **Kept categorical:** "
            + (", ".join(f"`{c}`" for c in info["categorical_kept"]) or L(lang, "—", "—"))
            + "\n"
            f"- **Numeric discretization:** `{info['bin_strategy']}`, bins=`{info['n_bins']}`\n"
            f"- **Binning fit on train only:** `{sample_info.get('binning_fit_train_only', True)}`\n"
            f"- **Row limit / sampling:** `{sample_info.get('row_sampling_enabled', False)}` "
            f"(rows used: {sample_info.get('rows_used_for_training_pipeline', 'N/A')} / {sample_info.get('rows_original', 'N/A')})"
        )
        if pipe_for_bins is not None:
            with st.expander(L(lang, "Bin edge ranges (numeric features)", "Khoảng giá trị từng **bin** (numeric)")):
                st.caption(
                    L(
                        lang,
                        "Edges are fit on **train**. A value landing in `bin_k` follows branch `bin_k` in ID3.",
                        "Ngưỡng (**edges**) fit trên **train**. Giá trị rơi vào `bin_k` → nhánh `bin_k` trong ID3.",
                    )
                )
                bin_map = _safe_bin_ranges(pipe_for_bins)
                if not bin_map:
                    st.write(L(lang, "No numeric features to show bin ranges.", "Không có cột numeric để hiển thị bin."))
                else:
                    chosen = st.selectbox(
                        L(lang, "Pick numeric feature", "Chọn cột **numeric**"),
                        options=sorted(bin_map.keys()),
                    )
                    st.write(f"**{chosen}**")
                    for line in bin_map[chosen]:
                        st.code(line)
    else:
        st.caption(L(lang, "Train once to populate this summary.", "Huấn luyện một lần để hiển thị phần này."))

    # --- Evaluation ---
    st.header(f"5. {L(lang, 'Evaluation', 'Đánh giá')}")
    if training_drift:
        st.warning(
            L(
                lang,
                "Training settings or the loaded dataset changed since the last Train. **Metrics below still reflect the last trained model** — click **Train ID3** to refresh them.",
                "Tham số hoặc bộ dữ liệu đã đổi so với lần Train trước. **Metrics bên dưới vẫn là của mô hình train gần nhất** — hãy bấm **Huấn luyện ID3** để cập nhật.",
            )
        )
    if lang == "Tiếng Việt":
        st.markdown(
            """
- **Accuracy:** tỷ lệ mẫu dự đoán đúng trên toàn bộ tập test.  
- **Precision (phishing):** trong các mẫu mô hình dự đoán là phishing, có bao nhiêu mẫu thực sự là phishing.  
- **Recall (phishing):** trong các mẫu phishing thực tế, mô hình phát hiện được bao nhiêu.  
- **F1-score:** trung bình điều hòa của Precision và Recall.  
- **Confusion matrix:** số lượng dự đoán đúng/sai theo từng lớp `legitimate` và `phishing`.
"""
        )
    else:
        st.markdown(
            """
- **Accuracy:** proportion of correct predictions on the test set.  
- **Precision (phishing):** among samples predicted as phishing, how many are truly phishing.  
- **Recall (phishing):** among truly phishing samples, how many are detected.  
- **F1-score:** harmonic mean of precision and recall.  
- **Confusion matrix:** counts correct/incorrect predictions for `legitimate` and `phishing`.
"""
        )
    if "model" not in st.session_state:
        st.caption(L(lang, "Train the model first.", "Hãy huấn luyện mô hình trước."))
    else:
        model: ID3Classifier = st.session_state["model"]
        X_test: pd.DataFrame = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        y_pred = model.predict(X_test)
        metrics = evaluate(y_test, y_pred)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        m2.metric("Precision (phishing)", f"{metrics['precision']:.4f}")
        m3.metric("Recall (phishing)", f"{metrics['recall']:.4f}")
        m4.metric("F1-score", f"{metrics['f1']:.4f}")
        st.caption(
            "Label mapping: `0` = legitimate, `1` = phishing"
            if lang != "Tiếng Việt"
            else "Quy ước nhãn: `0` = legitimate, `1` = phishing"
        )
        st.write("**Confusion matrix**")
        st.dataframe(confusion_matrix_df(y_test, y_pred), width="stretch")
        st.text("Classification report")
        st.text(report_string(y_test, y_pred))

    # --- Prediction & rule explanation ---
    st.header(f"6. {L(lang, 'Prediction & rule explanation', 'Dự đoán & giải thích luật')}")
    if training_drift and "model" in st.session_state:
        st.warning(
            L(
                lang,
                "Predictions and rules below use the **last trained** tree/preprocessor, not the current sliders until you retrain.",
                "Dự đoán và luật phía dưới dùng cây/preprocessor **đã train trước đó**, chưa theo thanh trượt hiện tại cho đến khi train lại.",
            )
        )
    if lang == "Tiếng Việt":
        st.markdown(
            """
- **Pick test row (default):** chọn một dòng trong tập **test**; so **prediction** với **true label**.  
- **Manual feature subset:** nhập vài **feature** quan trọng; phần còn lại lấy **default** từ **train**.  
- **Decision path:** các nhánh `feature = bin_x` hoặc **categorical** dọc cây **ID3**.  
- **IF–THEN rule:** cùng đường đi, viết dạng `IF ... THEN class = phishing | legitimate`.  
- **Explanation:** mô tả ngắn vì sao tới **leaf** đó.
"""
        )
    else:
        st.markdown(
            """
- **Pick test row (default):** choose one **test** row; compare **prediction** vs **true label**.  
- **Manual feature subset:** type a few important **features**; the rest use **train** defaults.  
- **Decision path:** branch tests `feature = bin_x` or a **categorical** value along the **ID3** tree.  
- **IF–THEN rule:** same path written as `IF ... THEN class = phishing | legitimate`.  
- **Explanation:** short note on why the leaf class was chosen.
"""
        )
    if "model" not in st.session_state or "test_df" not in st.session_state:
        st.caption(L(lang, "Train the model to unlock prediction.", "Huấn luyện mô hình để bật phần dự đoán."))
    else:
        pipe = st.session_state["pipe"]
        model = st.session_state["model"]
        test_df: pd.DataFrame = st.session_state["test_df"]

        _pick_test, _manual = "pick_test_row", "manual_features"
        mode = st.radio(
            L(lang, "Prediction mode", "Chế độ dự đoán"),
            [_pick_test, _manual],
            format_func=lambda k: (
                L(lang, "Pick test row (default)", "Chọn dòng tập test (mặc định)")
                if k == _pick_test
                else L(lang, "Manual feature subset", "Nhập tay một phần feature")
            ),
            horizontal=True,
        )

        if mode == _pick_test:
            idx = st.number_input(
                L(lang, "Test row index", "Chỉ số dòng trong tập test"), min_value=0, max_value=len(test_df) - 1, value=0
            )
            row = test_df.iloc[int(idx)]
            st.write(L(lang, "**Raw feature excerpt**", "**Trích đoạn feature (raw)**"))
            show_cols = [c for c in pipe.feature_columns if c in row.index][:20]
            st.dataframe(row[show_cols].to_frame().T, width="stretch")
            if st.button(L(lang, "Run prediction", "Chạy dự đoán")):
                pred, path = predict_test_row(model, pipe, row)
                if TARGET_COL in row.index:
                    st.write(
                        f"**{L(lang, 'True label', 'Nhãn đúng')}:** `{int(row[TARGET_COL])}` — **{label_to_display(int(row[TARGET_COL]))}**"
                    )
                st.write(f"**{L(lang, 'Prediction', 'Dự đoán')}:** **{format_prediction(pred)}**")
                st.write(f"**{L(lang, 'Decision path', 'Đường đi cây')}**")
                for step in path:
                    st.write(f"- {step[0]} = {step[1]} → {step[2]}")
                rule_text, natural_text = _format_path_rule(path, lang)
                st.write(f"**{L(lang, 'IF–THEN rule', 'Luật IF–THEN')}**")
                st.code(rule_text)
                st.markdown(f"**{L(lang, 'Explanation', 'Giải thích')}:** {natural_text}")
                st.write(L(lang, "**More IF–THEN rules (sample from tree)**", "**Thêm luật IF–THEN (mẫu từ cây)**"))
                for r in model.rules_to_text(max_rules=16):
                    st.code(r)

        else:
            st.caption(
                L(
                    lang,
                    "Only the fields below are required; missing values use train-time defaults.",
                    "Chỉ cần các trường dưới đây; phần còn lại lấy **default** từ train.",
                )
            )
            updates: dict = {}
            cols = st.columns(2)
            manual_fields = [f for f in MANUAL_PREDICTION_FEATURES if f in pipe.feature_columns]
            if pipe.default_raw_row is None:
                st.error(
                    L(
                        lang,
                        "Preprocessor has no train defaults; train the model again.",
                        "Thiếu giá trị mặc định từ train; hãy huấn luyện lại mô hình.",
                    )
                )
                st.stop()
            for i, fname in enumerate(manual_fields):
                with cols[i % 2]:
                    if fname == "TLD":
                        default_tld = str(pipe.default_raw_row.get(fname, "com"))
                        updates[fname] = st.text_input("TLD (e.g. com, de)", value=default_tld)
                    else:
                        dv = float(pipe.default_raw_row[fname])
                        updates[fname] = st.number_input(fname, value=dv, format="%.6f")
            if st.button(L(lang, "Run manual prediction", "Dự đoán thủ công")):
                # TLD as string; numeric as float
                manual_updates = {}
                for k, v in updates.items():
                    if k == "TLD":
                        manual_updates[k] = str(v).strip() or "com"
                    else:
                        manual_updates[k] = float(v)
                pred, path = predict_manual(model, pipe, manual_updates)
                st.write(f"**{L(lang, 'Prediction', 'Dự đoán')}:** **{format_prediction(pred)}**")
                st.write(f"**{L(lang, 'Decision path', 'Đường đi cây')}**")
                for step in path:
                    st.write(f"- {step[0]} = {step[1]} → {step[2]}")
                rule_text, natural_text = _format_path_rule(path, lang)
                st.write(f"**{L(lang, 'IF–THEN rule', 'Luật IF–THEN')}**")
                st.code(rule_text)
                st.markdown(f"**{L(lang, 'Explanation', 'Giải thích')}:** {natural_text}")
                for r in model.rules_to_text(max_rules=8):
                    st.code(r)

    st.divider()
    st.caption(L(lang, f"Project root: `{project_root()}`", f"Thư mục project: `{project_root()}`"))


if __name__ == "__main__":
    main()
