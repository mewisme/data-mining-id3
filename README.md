# Phishing URL Classifier (ID3 + Streamlit)

Ứng dụng học thuật phân loại URL thành **phishing** hoặc **legitimate** trên bộ dữ liệu **PhiUSIIL Phishing URL Dataset**.  
Mô hình sử dụng **ID3 tự cài đặt** (entropy + information gain), không dùng `DecisionTreeClassifier` để thay thế.

## Mục tiêu dự án

- Minh hoạ pipeline phân loại dữ liệu end-to-end: nạp dữ liệu -> tiền xử lý -> huấn luyện -> đánh giá -> dự đoán.
- Trực quan hóa các bước xử lý ngay trong giao diện Streamlit.
- Tập trung vào khả năng giải thích của cây quyết định (decision path, luật IF-THEN).

## Công nghệ chính

- **Python**
- **Streamlit** (UI)
- **Pandas / NumPy**
- **scikit-learn** (split dữ liệu, metric, hỗ trợ discretization)
- **Plotly** (biểu đồ tương tác)
- **Pytest** (kiểm thử)

## Dữ liệu đầu vào

- CSV theo schema PhiUSIIL.
- Cột mục tiêu: `label` (nhị phân).
- `FILENAME` là cột metadata, không dùng làm đặc trưng.

Kiểm tra schema được thực hiện khi nạp dữ liệu. Nếu thiếu cột bắt buộc hoặc nhãn không hợp lệ, hệ thống sẽ báo lỗi rõ ràng.

## Pipeline tiền xử lý

1. Tách `label` khỏi tập đặc trưng.
2. Bỏ các cột metadata / high-cardinality không phù hợp cho demo ID3 (`FILENAME`, `URL`, `Domain`, `Title`).
3. Xử lý missing values:
   - Numeric: điền theo median (fit trên train).
   - Categorical: dùng token missing / nhóm `OTHER`.
4. Biến đổi categorical:
   - Giữ top-N theo tập train.
   - Giá trị còn lại gom vào `OTHER`.
5. Discretization cột số bằng binning (`quantile` hoặc `uniform`), **fit trên train và áp dụng lại cho test/predict**.

## Mô hình và đánh giá

- Mô hình: **Custom ID3 Decision Tree**.
- Tiêu chí tách: **Entropy + Information Gain**.
- Train/test split ưu tiên stratified khi có thể.
- Chỉ số đánh giá:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix
  - Classification report

## Tính năng trong UI

- Tổng quan dữ liệu thô (shape, missing, class distribution, preview).
- Tóm tắt các bước preprocessing và cột bị loại.
- Trực quan train/test split.
- Dashboard pipeline theo từng giai đoạn.
- Huấn luyện với các tham số kiểm soát độ phức tạp cây.
- Dự đoán theo 2 chế độ:
  - Chọn dòng từ tập test.
  - Nhập tay đặc trưng (các trường còn thiếu dùng giá trị mặc định học từ train).
- Giải thích dự đoán:
  - Decision path
  - Luật IF-THEN
  - Mô tả ngôn ngữ tự nhiên cho nhánh đã đi.

## Cài đặt

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
# source venv/bin/activate

pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
streamlit run app.py
```

Sau khi chạy, tải file CSV trong sidebar (hoặc dùng dữ liệu mặc định nếu có sẵn trong thư mục `data/`).

## Chạy kiểm thử

```bash
python -m pytest
```

## Cấu hình qua biến môi trường

- `PHISHING_DEBUG_ERRORS` (mặc định: `false`)
  - `true`: hiển thị lỗi chi tiết để debug.
  - `false`: hiển thị lỗi ngắn gọn, an toàn cho người dùng.
- `PHISHING_DEFAULT_ROW_LIMIT` (mặc định: `8000`)
  - Thiết lập giới hạn số dòng mặc định trong Training Controls.
  - `0` = dùng toàn bộ dữ liệu.

## Cấu trúc thư mục

```text
phishing-url/
  app.py
  requirements.txt
  README.md
  data/
  src/
    data_loader.py
    preprocessing.py
    id3.py
    evaluation.py
    predictor.py
    utils.py
    services/
      data_service.py
      training_service.py
    ui/
      common.py
      charts.py
      sections/
```

## Hạn chế hiện tại

- ID3 có thể chậm trên dữ liệu lớn.
- Chất lượng mô hình phụ thuộc mạnh vào chiến lược rời hoá (`n_bins`, `quantile/uniform`).
- Dữ liệu categorical chưa thấy trong train sẽ đi theo cơ chế fallback của cây.

## Hướng phát triển

- Thử các kỹ thuật discretization có giám sát.
- Bổ sung cross-validation.
- Tối ưu hiệu năng huấn luyện/preprocessing.
- Mở rộng phần giải thích mô hình cho mục tiêu giảng dạy.

## Ghi chú

Dự án phục vụ mục đích học tập trong môn phân loại dữ liệu / an toàn thông tin.
