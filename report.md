# Báo Cáo Thực Hành Machine Learning: Dự Đoán Chi Phí Bảo Hiểm Y Tế

## 1. Giới thiệu bài toán
Dự án này thực hiện xây dựng một mô hình Machine Learning (Regression) để dự đoán chi phí bảo hiểm y tế (`charges`) dựa trên các thông tin cá nhân của người tham gia bảo hiểm. Mục tiêu là vận dụng quy trình từ khám phá, tiền xử lý dữ liệu đến huấn luyện, tìm ra mô hình có khả năng dự báo chính xác nhất và phân tích các nguyên nhân cốt lõi tác động lên mô hình.

## 2. Mô tả dataset
Dataset được sử dụng là **Medical Insurance Cost**. Dữ liệu bao gồm 1338 mẫu và 7 biến, trong đó:
- **age**: Tuổi người tham gia bảo hiểm (số nguyên).
- **sex**: Giới tính (nam/nữ).
- **bmi**: Chỉ số khối cơ thể (số thực).
- **children**: Số trẻ em/người phụ thuộc được bảo hiểm chi trả.
- **smoker**: Tình trạng hút thuốc (có/không).
- **region**: Khu vực sinh sống tại Mỹ (northeast, northwest, southeast, southwest).
- **charges** (Biến mục tiêu): Chi phí y tế cá nhân được thanh toán (số liên tục).

## 3. Khám phá dữ liệu (EDA)
Dựa trên kết quả phân tích (chi tiết tại `notebook/EDA.ipynb`), có một số insight nổi bật:
- **Biến mục tiêu `charges`**: Có phân phối lệch phải (right-skewed) rất mạnh. Đa số chi phí tập trung ở mức thấp, nhưng có một lượng đuôi dài kéo đến vùng chi phí rất cao.
- **Biến `smoker`**: Có tác động rõ rệt nhất đến `charges`. Người hút thuốc thường phải trả chi phí cao hơn hẳn so với người không hút thuốc.
- **Tương tác `smoker` và `bmi`**: Nhóm người hút thuốc và có `bmi` cao (béo phì) đối mặt với mức chi phí lớn nhất.
- Dữ liệu không có giá trị bị thiếu (missing values) nhưng có 1 quan sát bị trùng lặp.

## 4. Tiền xử lý dữ liệu
Dữ liệu được chuẩn bị cẩn thận để tránh rò rỉ (Data Leakage) theo các bước:
1. **Loại bỏ trùng lặp**: Xóa 1 dòng lặp lại trong quá trình đọc dữ liệu.
2. **Chia Train/Test Split**: Thực hiện chia theo tỷ lệ 80/20 với `random_state=42` để đảm bảo tính lặp lại.
3. **Mã hóa (Encoding)**: Sử dụng `OneHotEncoder(drop='first')` trên các biến phân loại (`sex`, `smoker`, `region`).
4. **Chuẩn hóa (Scaling)**: Sử dụng `StandardScaler` cho các biến số (`age`, `bmi`, `children`) để tối ưu hiệu suất cho các thuật toán tính toán khoảng cách (KNN, SVR).
Quá trình Encode và Scale chỉ fit trên tập Train, sau đó transform cho tập Test thông qua `ColumnTransformer` và `Pipeline`.

---

## 5. Kết Quả Huấn Luyện Các Mô Hình Cơ Bản

Dưới đây là phần tổng hợp và phân tích kết quả dựa trên các chỉ số thu thập được từ quá trình huấn luyện các mô hình Machine Learning dự đoán chi phí bảo hiểm y tế (`charges`).

### Bảng So Sánh Kết Quả Bắt Buộc

| Mô hình | MAE | MSE | RMSE | R2 Score |
|---------|-----|-----|------|----------|
| **Linear Regression** | 4177.04 | 35,478,020 | 5956.34 | 0.8069 |
| **Decision Tree** | 2787.71 | 34,865,120 | 5904.66 | 0.8102 |
| **Random Forest** | **2657.61** | **22,316,360** | **4724.01** | **0.8785** |
| **K-Nearest Neighbors** | 4494.19 | 65,004,150 | 8062.51 | 0.6462 |
| **Support Vector Regressor** | 9261.82 | 208,257,100 | 14431.11 | -0.1333 |

---

## 6. Phân Tích & So Sánh Mô Hình

### Mô hình cho kết quả tốt nhất
Trong phạm vi các mô hình cơ bản, **Random Forest Regressor** đạt kết quả tốt nhất.

### Tiêu chí lựa chọn mô hình
Quyết định này được đưa ra dựa trên hai chỉ số đánh giá:
1. **R2 Score (0.8785):** Mô hình giải thích được khoảng 87.85% sự biến thiên của chi phí y tế.
2. **MAE (2657.61):** Sai số tuyệt đối trung bình của dự đoán là 2,657 so với giá trị thực tế.

### Vì sao Random Forest hoạt động hiệu quả?
- **Ensemble Learning:** Random Forest kết hợp nhiều Decision Trees, qua đó giảm thiểu tính high variance của một cây quyết định đơn lẻ, giúp mô hình generalize tốt hơn.
- **Khả năng nắm bắt quan hệ phi tuyến:** Chi phí y tế chịu ảnh hưởng bởi nhiều yếu tố tương tác phức tạp (ví dụ: tác động kết hợp giữa việc hút thuốc và chỉ số BMI cao). Random Forest xử lý các tương tác phi tuyến này hiệu quả hơn so với Linear Regression.

### Tại sao Support Vector Regressor (SVR) đạt hiệu suất kém?
Mô hình SVR có hiệu suất thấp (R2 âm: -0.1333, MAE: 9261.82). Nguyên nhân chủ yếu xuất phát từ đặc tính của thuật toán khi áp dụng trên biến mục tiêu chưa được chuẩn hóa:
1. **Lack of Scale Invariance:** Khác với Random Forest (chia cắt không gian đặc trưng) hay Linear Regression (tối ưu hóa hệ số), SVR tính toán khoảng cách hình học giữa các điểm dữ liệu. Việc không scale biến mục tiêu `y` (dao động từ 1,000 đến 60,000) làm giảm độ chính xác của hàm khoảng cách.
2. **Tham số $\epsilon$:** SVR xây dựng một ống sai số $\epsilon$ xung quanh các điểm dữ liệu. Với phương sai lớn và phân phối lệch phải của `y`, giá trị mặc định của $\epsilon$ trở nên quá nhỏ, khiến thuật toán không thể xác định các support vectors một cách tối ưu.

### Đánh giá hiện tượng Overfitting và Underfitting
Dựa trên phân tích **Learning Curve**:
- **Decision Tree:** Thể hiện Overfitting. Train Error tiến về 0, nhưng Validation Error duy trì ở mức cao.
- **Support Vector Regressor:** Thể hiện Underfitting. Error rất cao ở cả tập Train và tập Validation.
- **Random Forest:** Có mức độ Overfitting nhẹ (đặc trưng của các mô hình cây), biểu hiện qua khoảng cách giữa Train Error và Validation Error. Dù vậy, mô hình vẫn đạt Validation Error thấp nhất.

### Những khó khăn gặp phải trong quá trình thực hiện
1. Biến mục tiêu `charges` phân phối right-skewed, làm giảm hiệu suất dự đoán của các mô hình, đặc biệt tại các vùng giá trị cao.
2. Cần cẩn thận trong việc thiết kế `ColumnTransformer` để đảm bảo thao tác fit/transform chính xác trên các tập dữ liệu, ngăn chặn Data Leakage triệt để đối với các biến phân loại và biến liên tục.

---

## 7. Kết Luận Và Hướng Cải Thiện (Bonus Pipeline)

Nhằm khắc phục các hạn chế đã phân tích, đặc biệt là tính right-skewed của `charges` và hiệu suất thấp của SVR, một Pipeline bổ sung (`src/bonus_pipeline.py`) đã được đề xuất và thực hiện.

**Hướng cải thiện áp dụng:**
1. **TransformedTargetRegressor:** Áp dụng `np.log1p` (Log Transform) cho biến mục tiêu $y$ trong quá trình huấn luyện và `np.expm1` để khôi phục kết quả dự đoán. Việc này đưa biến mục tiêu về dạng gần phân phối chuẩn, hỗ trợ tốt cho các mô hình tính toán độ lỗi theo khoảng cách.
2. **Gradient Boosting:** Bổ sung các mô hình tree-based tiên tiến hơn là **XGBoost** và **LightGBM** để nắm bắt dữ liệu phức tạp tốt hơn.

### Kết Quả Sau Cải Thiện

| Mô hình | MAE | MSE | RMSE | R2 Score |
|---------|-----|-----|------|----------|
| **Log Transformed Linear Regression** | 3755.92 | 43,158,359 | 6569.50 | 0.7181 |
| **Log Transformed SVR** | 2518.39 | 24,286,851 | 4928.16 | 0.8413 |
| **XGBoost** | 2663.22 | 21,639,520 | 4651.82 | 0.8586 |
| **LightGBM** | **2344.65** | **17,027,330** | **4126.41** | **0.8887** |

### Kết luận
1. **Cải thiện mạnh mẽ hiệu suất SVR:** Khi áp dụng Log Transform để giảm độ lớn và biến đổi phân phối của biến mục tiêu, R2 Score của SVR tăng từ **-0.133** lên **0.841**. Điều này xác nhận nguyên nhân kém cỏi trước đó do giới hạn của hàm khoảng cách.
2. **Mô hình tối ưu nhất:** LightGBM kết hợp với Log Transform đạt hiệu suất cao nhất dự án với R2 Score **0.8887** và MAE **2344**. Do đó, LightGBM là thuật toán tốt nhất và nên được chọn làm mô hình cuối cùng để dự báo chi phí bảo hiểm y tế trên bộ dữ liệu này.
