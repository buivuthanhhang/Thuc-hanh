# ⚖️ Financial Statement Fraud Detection Web App (Beneish M-Score & XGBoost)

Ứng dụng web tương tác xây dựng trên nền tảng **Streamlit**, tích hợp đồng thời hai thuật toán học máy mạnh mẽ: **Hồi quy Logistic (Logistic Regression)** và **XGBoost Classifier (Extreme Gradient Boosting)** kết hợp bộ **8 chỉ số Beneish M-Score** để dự báo và phát hiện sớm nguy cơ gian lận Báo cáo Tài chính (BCTC) của doanh nghiệp.

Ứng dụng được thiết kế tối ưu cho Kiểm toán viên, Nhà phân tích tài chính, Chuyên viên quản trị rủi ro và Nhà đầu tư chứng khoán.

---

## 🌟 Các Tính Năng Nổi Bật

1. **Khám Phá Dữ Liệu Tương Tác (Data Exploration):**
   - Hỗ trợ dữ liệu mặc định (`MScore_data.csv`) hoặc tải lên file CSV tùy biến.
   - Thống kê mô tả (Mean, Std, Min, Max), phân bổ nhãn gian lận và biểu đồ tương quan nhiệt (Correlation Matrix).

2. **So Sánh Đa Mô Hình (Model Comparison: Logistic Regression vs. XGBoost):**
   - Huấn luyện song song cả hai mô hình trên cùng một tập Train / Test.
   - Bảng so sánh hiệu năng trực tiếp giữa hai mô hình: Accuracy, Precision, Recall, F1-Score, Specificity, FPR, FNR (tự động gắn cờ mô hình vượt trội).
   - So sánh trực quan Ma trận nhầm lẫn (Confusion Matrix) cạnh nhau.
   - Phân tích trọng số Beta & Tỷ số Odds Ratio của Logistic Regression và biểu đồ **Feature Importance (Tầm quan trọng đặc trưng)** của XGBoost.
   - **Xuất báo cáo đa mô hình toàn diện ra file Excel đa trang (`.xlsx`)**.

3. **Công Cụ Dự Báo Doanh Nghiệp Đa Mô Hình (Multi-Model Fraud Prediction):**
   - **Dự báo đơn lẻ (Single Prediction):** Nhập 8 chỉ số của 1 doanh nghiệp (có nút nạp mẫu nhanh), hiển thị đồng thời 2 đồng hồ đo xác suất (Gauge charts) của cả Logistic Regression và XGBoost, kết luận **Đồng thuận rủi ro** (Rủi ro rất cao, Phân kỳ hay An toàn), và đối chiếu ngưỡng cảnh báo của GS. Beneish.
   - **Dự báo hàng loạt (Batch CSV Upload):** Tải lên danh sách nhiều doanh nghiệp, hệ thống tự động chấm điểm xác suất từ cả 2 mô hình, phân loại rủi ro tổng hợp và cho phép xuất kết quả về file CSV.

4. **Cẩm Nang & Phương Pháp Luận (Beneish & Machine Learning Guide):**
   - Chi tiết công thức, ý nghĩa kinh tế của 8 chỉ số Beneish.
   - So sánh chuyên sâu giữa mô hình tuyến tính (Logistic Regression) và mô hình phi tuyến tính tập hợp cây (XGBoost) trong kiểm toán.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
├── app.py                             # Mã nguồn chính của ứng dụng Streamlit (Logistic + XGBoost)
├── requirements.txt                   # Danh sách các thư viện phụ thuộc (bao gồm xgboost)
├── MScore_data.csv                    # Tập dữ liệu mẫu 8 chỉ số Beneish & nhãn FRAUD_FLAG
├── README.md                          # Tài liệu hướng dẫn sử dụng & triển khai
└── logistic_regression_mscore_colab.py # File script gốc từ Google Colab (để tham chiếu)
```

---

## 💻 Hướng Dẫn Cài Đặt & Chạy Trên Máy Cá Nhân (Local)

### Bước 1: Chuẩn bị môi trường Python
Khuyến nghị sử dụng Python 3.9 đến 3.11.

Mở **Terminal** (trên macOS/Linux) hoặc **Command Prompt / PowerShell** (trên Windows), di chuyển vào thư mục dự án:
```bash
cd "/duong/dan/den/thu/muc/du/an"
```

### Bước 2: Tạo và kích hoạt môi trường ảo (Khuyến nghị)
- **Trên macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- **Trên Windows:**
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```

### Bước 3: Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### Bước 4: Khởi chạy ứng dụng Web
```bash
streamlit run app.py
```
Sau khi chạy lệnh, trình duyệt web sẽ tự động mở địa chỉ: `http://localhost:8501`.

---

## 🚀 Hướng Dẫn Đẩy Lên GitHub

Để triển khai lên Streamlit Cloud, bạn cần đưa toàn bộ mã nguồn lên một kho chứa (repository) trên GitHub.

### Bước 1: Tạo Repository mới trên GitHub
1. Truy cập [https://github.com](https://github.com) và đăng nhập.
2. Nhấn nút **New** (Tạo repository mới).
3. Đặt tên repository, ví dụ: `bctc-fraud-detector`.
4. Chọn chế độ **Public** (Công khai để triển khai Streamlit Cloud miễn phí).
5. Nhấn **Create repository**.

### Bước 2: Khởi tạo và đẩy code từ máy tính lên GitHub
Mở Terminal tại thư mục dự án và thực hiện tuần tự các lệnh sau:

```bash
# 1. Khởi tạo git local
git init

# 2. Thêm toàn bộ file vào git tracking
git add app.py requirements.txt README.md MScore_data.csv

# 3. Commit code
git commit -m "Cập nhật ứng dụng Streamlit dự báo gian lận BCTC với Logistic Regression và XGBoost"

# 4. Đổi tên nhánh chính sang main
git branch -M main

# 5. Liên kết tới repository GitHub của bạn (thay bằng URL repo của bạn)
git remote add origin https://github.com/<USERNAME-CUA-BAN>/bctc-fraud-detector.git

# 6. Đẩy mã nguồn lên GitHub
git push -u origin main
```

---

## ☁️ Hướng Dẫn Triển Khai (Deploy) Lên Streamlit Community Cloud

Streamlit Community Cloud cho phép bạn host ứng dụng web hoàn toàn **miễn phí** trực tiếp từ GitHub:

1. **Đăng nhập Streamlit Cloud:**
   - Truy cập [https://share.streamlit.io](https://share.streamlit.io).
   - Đăng nhập bằng tài khoản GitHub vừa đẩy code.

2. **Tạo App Mới:**
   - Nhấn vào nút **Create app** (hoặc **New app**).
   - Chọn repository của bạn: `<USERNAME-CUA-BAN>/bctc-fraud-detector`.
   - **Branch:** Chọn `main`.
   - **Main file path:** Điền `app.py`.

3. **Nhấn Deploy!**
   - Hệ thống Streamlit Cloud sẽ tự động đọc file `requirements.txt` (cài đặt `scikit-learn`, `xgboost`, `plotly`, `pandas`,...), xây dựng môi trường và khởi động ứng dụng trong khoảng 1-2 phút.
   - Khi hoàn tất, bạn sẽ nhận được đường dẫn công khai (Public URL) để chia sẻ cho đồng nghiệp hoặc báo cáo khoa học.

---

## 📊 Định Dạng Dữ Liệu Đầu Vào (Input Format)

File CSV chuẩn cần có 8 cột chỉ số:
- `DSRI`: Days Sales in Receivables Index (Số ngày thu tiền khách hàng)
- `GMI`: Gross Margin Index (Tỷ suất lợi nhuận gộp)
- `AQI`: Asset Quality Index (Chất lượng tài sản)
- `SGI`: Sales Growth Index (Tăng trưởng doanh thu)
- `DEPI`: Depreciation Index (Tỷ lệ khấu hao)
- `SGAI`: Sales, General & Administrative expenses Index (Chi phí bán hàng & quản lý)
- `TATA`: Total Accruals to Total Assets (Dồn tích kế toán trên tổng tài sản)
- `LVGI`: Leverage Index (Đòn bẩy tài chính)
- `FRAUD_FLAG`: Nhãn mục tiêu (chỉ cần cho file huấn luyện):
  - `0`: Báo cáo tài chính trung thực / Không gian lận
  - `1`: Báo cáo tài chính có gian lận / Thao túng lợi nhuận

---

## 📜 Giấy Phép & Tác Quyền
Dự án được phát triển phục vụ mục đích nghiên cứu, học tập và ứng dụng thực tiễn trong phân tích dữ liệu tài chính kế toán.
