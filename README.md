# Dự Án Dự Đoán Chi Phí Bảo Hiểm Y Tế

Dự án này là Bài tập thực hành cá nhân (Môn học: Nhập môn Khoa học Dữ liệu), giải quyết bài toán Regression nhằm dự báo chi phí bảo hiểm y tế dựa trên dữ liệu thông tin cá nhân.

## 1. Cấu trúc thư mục

```text
├── README.md                # Hướng dẫn chạy cài đặt và thực thi code
├── report.pdf               # Báo cáo chi tiết quá trình và kết quả bản PDF
├── requirements.txt         # Danh sách các thư viện cần thiết
├── src/                     # Thư mục chứa mã nguồn Python chính
│   ├── config.py            # Cấu hình đường dẫn dự án
│   ├── data_processing.py   # Các hàm tiền xử lý dữ liệu
│   ├── model_training.py    # Các hàm khởi tạo mô hình
│   ├── evaluation.py        # Các hàm đánh giá và lưu kết quả
│   ├── main_pipeline.py     # Pipeline chạy 5 mô hình cơ bản 
│   └── bonus_pipeline.py    # Pipeline cải tiến (Log Transform, XGBoost, LightGBM)
├── notebook/
│   └── EDA.ipynb            # Notebook phân tích trực quan dữ liệu
├── data/
│   └── Medical_Insurance_Cost.csv # Tập dữ liệu gốc
└── outputs/                 # Thư mục chứa kết quả sinh ra tự động
    ├── figures/             # Các biểu đồ đánh giá mô hình
    └── results.csv          # Bảng tổng hợp kết quả (MAE, MSE, R2...)
```

## 2. Hướng dẫn thiết lập môi trường

Đảm bảo bạn đang sử dụng **Python 3.10** hoặc mới hơn. Cài đặt các thư viện cần thiết bằng lệnh:

```bash
pip install -r requirements.txt
```
*(Lưu ý: Dự án có đính kèm file `pyproject.toml` và `uv.lock`. Nếu bạn dùng bộ quản lý gói `uv`, bạn có thể khởi tạo môi trường trực tiếp bằng lệnh `uv sync`).*

## 3. Hướng dẫn chạy mã nguồn

### Phân tích Khám phá Dữ liệu (EDA)
Để xem các đánh giá trực quan về độ tương quan, phân phối của dữ liệu và các insight quan trọng, hãy mở file `notebook/EDA.ipynb` (có thể dùng Jupyter Notebook hoặc VS Code).

### Chạy quy trình huấn luyện cơ bản (Main Pipeline)
Thực thi lệnh sau để bắt đầu tiền xử lý và chạy 5 mô hình cơ bản (Linear Regression, Decision Tree, Random Forest, KNN, SVR):
```bash
python src/main_pipeline.py
```
- Bảng đánh giá sẽ được in trực tiếp ra màn hình.
- File CSV lưu kết quả nằm ở `outputs/results.csv`.
- Các biểu đồ Actual vs Predicted, Residual, Feature Importance được lưu ở thư mục `outputs/figures/`.

### Chạy quy trình cải tiến mở rộng (Bonus Pipeline)
Thực thi lệnh sau để xem hiệu quả của việc kết hợp Log-Transform trên biến mục tiêu (để giải quyết phân phối lệch) và huấn luyện trên các mô hình Gradient Boosting mạnh mẽ (XGBoost, LightGBM):
```bash
python src/bonus_pipeline.py
```
Kết quả của pipeline cải thiện được lưu tại `outputs/bonus_results.csv`.
