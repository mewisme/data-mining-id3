# Xây dựng ứng dụng phân loại URL phishing bằng thuật toán ID3 sử dụng Python

Ứng dụng demo cho môn **phân loại dữ liệu**: phân loại nhị phân **phishing** vs **legitimate** trên bộ **PhiUSIIL Phishing URL Dataset**, dùng **cây quyết định ID3 tự cài đặt** (entropy, information gain, đệ quy) và giao diện **Streamlit**.

## Tại sao chọn PhiUSIIL?

- Bộ dữ liệu công khai, phổ biến trong bài toán URL/web phishing, có nhiều đặc trưng đã được tính sẵn (độ dài URL, TLD, tỷ lệ ký tự, v.v.).
- Phù hợp môn học: có thể minh họa **chuỗi xử lý**: đọc CSV → tiền xử lý → huấn luyện → đánh giá → dự đoán.

## Tại sao dùng ID3?

- **ID3** là thuật toán cây quyết định kinh điển dựa trên **entropy** và **information gain**, phù hợp đề tài “phân loại dữ liệu”.
- Lưu ý: **không** dùng `sklearn.tree.DecisionTreeClassifier` để “đóng vai” ID3 — trong repo này, cây được **tự implement**; `scikit-learn` chỉ dùng cho `train_test_split`, metric, và tiện ích binning (`KBinsDiscretizer`).
- Trong UI có ghi rõ: **Model: Custom ID3 decision tree**; tiêu chí tách là **entropy + information gain**.

## Schema dữ liệu (tóm tắt)

- **Target:** `label` — nhị phân (trong file mẫu: `0` = legitimate, `1` = phishing).
- **Không dùng làm đặc trưng:** `FILENAME` (metadata/định danh file).
- **Đặc trưng:** các cột còn lại theo danh sách chuẩn PhiUSIIL (URL, URLLength, Domain, …, NoOfExternalRef).

Khi đọc CSV, `validate_schema` yêu cầu đủ **`label`** và **toàn bộ** tên cột trong `FEATURE_COLUMNS` (`src/utils.py`). Cột **`FILENAME`** là tuỳ chọn (nếu có sẽ bị bỏ khi tiền xử lý).

**Nhãn mục tiêu:** chỉ chấp nhận nhị phân **`0`/`1`** (hoặc float tương đương), hoặc chuỗi trong tập token cố định (ví dụ `phishing` / `legitimate`). Giá trị lạ hoặc thiếu sẽ báo lỗi rõ ràng, không gán âm thầm vào lớp `0`.

## Cột bị loại / thiết kế tiền xử lý

| Thành phần | Xử lý |
|------------|--------|
| `FILENAME` | Luôn bỏ (không phải tín hiệu học). |
| `URL`, `Domain`, `Title` | **Luôn bỏ** trong pipeline demo: độ cardinal cực cao; giữ lại sẽ làm cây ID3 phình to, và ép kiểu số sẽ làm sai ngữ nghĩa. |
| `TLD` (và mọi cột khai báo trong `CATEGORICAL_FEATURES`) | **Hạng mục**: top-N trên **tập train** mỗi cột, còn lại gộp `OTHER`. |
| Cột số / đếm / tỷ lệ còn lại | **Rời hóa** bằng bin (quantile hoặc uniform), **chỉ fit trên train**. |
| Giá trị thiếu | Số: median (train); hạng mục: `__MISSING__` / `OTHER`. |

## Kiểm thử (tối thiểu)

```bash
python -m pytest
```

## Cài đặt

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
```

Khi chạy trên Streamlit deployment, ứng dụng ở chế độ **upload-only**: người dùng cần tải lên file CSV PhiUSIIL trong sidebar.

## Chạy ứng dụng

```bash
streamlit run app.py
```

- Cần **upload CSV** trong sidebar để bắt đầu sử dụng.

## Trực quan pipeline trong Streamlit

UI đã được nâng cấp để hiển thị rõ toàn bộ pipeline xử lý dữ liệu theo từng bước thực tế (không dùng dữ liệu giả):

- **Raw Data Overview**: shape, số cột, phân phối lớp, missing values, kiểu dữ liệu, preview dữ liệu thô.
- **Feature Selection / Dropped Columns**: cột nào được dùng, cột nào bị bỏ, và lý do.
- **Target Interpretation**: ánh xạ nhãn gốc -> nhãn nhị phân `0/1`, kèm lỗi validation rõ ràng nếu nhãn không hợp lệ.
- **Train/Test Split Visualization**: số dòng train/test, tỉ lệ split, random state, phân phối lớp trước và sau split.
- **Missing Value Handling**: so sánh trước/sau và chiến lược điền thiếu theo từng cột.
- **Numeric Binning / Discretization**: cột số nào được rời hóa, số bins thực tế, ngưỡng bins, ví dụ giá trị gốc -> `bin_k`.
- **Categorical Transformation**: top-N theo train, xử lý `OTHER`, và xử lý missing (`__MISSING__`).
- **Final Model Input Preview**: xem trực tiếp `X_train`/`X_test` sau biến đổi.
- **Single Row Data Journey**: theo dõi một dòng dữ liệu qua các stage từ raw -> transformed -> prediction path.
- **Feature Space Explorer (2D/3D có chọn lọc)**: scatter tương tác để quan sát phân tách lớp theo feature; 3D chỉ dùng cho khám phá không gian đặc trưng, không dùng cho biểu đồ đếm.

Ngoài bảng chi tiết, phần lớn bước preprocessing đã có chart tương tác (Plotly) để giảm “text/table-heavy” và tăng tính demo/giảng dạy.

## Cấu trúc thư mục (hiện tại)

```
project_root/
  app.py                 # Streamlit UI
  requirements.txt
  README.md
  data/
    PhiUSIIL_Phishing_URL_Dataset.csv
  src/
    data_loader.py       # Đọc CSV, kiểm tra schema, resolve default path
    preprocessing.py     # Tiền xử lý + rời hóa cho ID3 (fit trên train)
    id3.py               # ID3 (entropy, gain, cây)
    evaluation.py        # Accuracy, precision, recall, F1, confusion matrix, report
    predictor.py         # Dự đoán + giải thích đường đi
    utils.py             # Hằng số schema, cột mục tiêu, tiện ích hiển thị
    services/
      data_service.py     # Nạp dữ liệu từ upload/default CSV cho UI
      training_service.py # Orchestrate train/test split, fit pipeline, train model
    ui/
      common.py           # State/UI helpers, render section, i18n helper
      charts.py           # Các biểu đồ dùng trong dashboard
      sections/
        intro.py
        dataset_overview.py
        training_controls.py
        preprocess_summary.py
        pipeline_section.py
        evaluation_section.py
        prediction_section.py
```

## Đầu vào / đầu ra

**Đầu vào:** file CSV PhiUSIIL; (tuỳ chọn) chọn dòng trên tập test; hoặc nhập tay một phần đặc trưng (các trường còn lại lấy mặc định từ train).

**Đầu ra:** nhãn dự đoán (phishing / legitimate), metrics, confusion matrix, classification report, **đường đi trên cây**, **luật IF-THEN dễ đọc**, và mô tả ngôn ngữ tự nhiên cho nhánh đã đi qua.

## Luồng chạy trong app

- Sidebar hỗ trợ 2 nguồn dữ liệu: **upload CSV** hoặc dùng file mặc định trong `data/` nếu tồn tại.
- UI theo kiến trúc module: mỗi section (`intro`, `overview`, `training`, `pipeline`, `evaluation`, `prediction`) render tách file để dễ bảo trì.
- Khi bấm train, `run_training` trong `src/services/training_service.py` sẽ:
  - chuẩn hóa target `label`,
  - split train/test (ưu tiên stratified),
  - fit preprocessing **chỉ trên train**,
  - huấn luyện mô hình ID3 custom,
  - trả artifact lưu trong `st.session_state` để tái sử dụng cho các section.

## Cách tiền xử lý hoạt động

- Tách `label` làm target.
- Bỏ `FILENAME` khỏi tập đặc trưng.
- Luôn bỏ `URL`, `Domain`, `Title` (high-cardinality; không hỗ trợ trong demo này).
- Giữ `TLD` dưới dạng categorical (top-N, còn lại `OTHER`).
- Cột số được rời hóa bằng bin (`quantile` hoặc `uniform`) và **fit chỉ trên train**, sau đó tái sử dụng cho test/predict.
- UI hiển thị tóm tắt tiền xử lý gồm: cột bỏ, chiến lược binning, số bins, train-only fitting, trạng thái row limit/sampling. Nếu đổi tham số sau khi train, UI cảnh báo metrics/cây hiện tại là của **lần train trước** cho tới khi train lại.

## Cách dự đoán hoạt động

- **Mode A (mặc định):** chọn một dòng trong tập test, dự đoán và đối chiếu true label.
- **Mode B:** nhập tay một tập nhỏ đặc trưng quan trọng; các đặc trưng còn thiếu lấy theo giá trị mặc định học từ train.
- Phần giải thích hiển thị:
  - đường đi kỹ thuật qua các node,
  - 1 luật dạng `IF ... THEN class = ...`,
  - mô tả ngôn ngữ tự nhiên cho người học.

## Hạn chế

- ID3 trên dữ liệu lớn vẫn có thể **chậm** — dùng **giới hạn số dòng**, **max depth**, **min samples split** trong UI.
- Rời hóa bằng bin là xấp xỉ; kết quả phụ thuộc `n_bins` và chiến lược bin.
- Dự đoán với giá trị hạng mục **chưa thấy** trên nhánh cây: dùng **nhãn đa số** tại nút đó (fallback).

## Hướng cải tiến

- Thử thêm rời hóa có giám sát (MDLPC, v.v.), hoặc chọn đặc trưng trước khi học cây.
- Tối ưu hiện thực ID3 (numpy, sparse) hoặc giới hạn số giá trị phân nhánh.
- Thêm đánh giá cross-validation.

## Reliability & configuration

Project now hardens common failure paths (data loading, split edge-cases, and partial UI state). By default, user-facing errors stay concise.

Environment variables:

- `PHISHING_DEBUG_ERRORS` (default: `false`)
  - `true`: show detailed exception traces/messages in UI for debugging.
  - `false`: show concise user-safe errors only.
- `PHISHING_DEFAULT_ROW_LIMIT` (default: `8000`)
  - Controls default row limit shown in Training Controls.
  - `0` means use all rows.

## Lightweight performance profiling

If preprocessing/training feels slow, profile before optimizing:

```bash
python -m cProfile -o train_profile.prof app.py
```

Then inspect hot functions:

```bash
python -m pstats train_profile.prof
```

## Tác giả / môn học

Dự án phục vụ mục đích học tập (data classification / cybersecurity).
