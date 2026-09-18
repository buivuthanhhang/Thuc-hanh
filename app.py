# -*- coding: utf-8 -*-
"""
Web App: Dự báo Gian lận Báo cáo Tài chính (Financial Statement Fraud Detection)
Mô hình: Hồi quy Logistic (Logistic Regression) kết hợp 8 biến chỉ số Beneish M-Score
Nền tảng: Streamlit
"""

import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ==============================================================================
# 1. CẤU HÌNH TRANG WEB & THEME
# ==============================================================================
st.set_page_config(
    page_title="Dự Báo Gian Lận BCTC | Beneish M-Score",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tùy biến CSS tăng tính thẩm mỹ & chuyên nghiệp
st.markdown("""
<style>
    /* Kiểu font và tiêu đề */
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    /* Card thông số KPI */
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1F2937;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    /* Khung cảnh báo */
    .risk-high {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
        padding: 15px;
        border-radius: 8px;
        color: #991B1B;
    }
    .risk-low {
        background-color: #DCFCE7;
        border-left: 5px solid #22C55E;
        padding: 15px;
        border-radius: 8px;
        color: #166534;
    }
</style>
""", unsafe_allow_html=True)

# 8 biến Beneish M-score chuẩn
FEATURES = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "TATA", "LVGI"]
TARGET = "FRAUD_FLAG"

# Thông tin diễn giải chi tiết 8 biến
FEATURE_DESCRIPTIONS = {
    "DSRI": "Days Sales in Receivables Index (Chỉ số số ngày thu tiền khách hàng)",
    "GMI": "Gross Margin Index (Chỉ số tỷ suất lợi nhuận gộp)",
    "AQI": "Asset Quality Index (Chỉ số chất lượng tài sản)",
    "SGI": "Sales Growth Index (Chỉ số tăng trưởng doanh thu)",
    "DEPI": "Depreciation Index (Chỉ số tỷ lệ khấu hao)",
    "SGAI": "Sales, General & Admin expenses Index (Chỉ số chi phí bán hàng & quản lý)",
    "TATA": "Total Accruals to Total Assets (Tỷ lệ dồn tích kế toán trên tổng tài sản)",
    "LVGI": "Leverage Index (Chỉ số đòn bẩy tài chính)"
}

# Ngưỡng cảnh báo kinh nghiệm của Beneish (nếu > ngưỡng thì có dấu hiệu nghi vấn)
BENCHMARK_THRESHOLDS = {
    "DSRI": 1.031,
    "GMI": 1.014,
    "AQI": 1.039,
    "SGI": 1.134,
    "DEPI": 1.001,
    "SGAI": 1.054,
    "TATA": 0.018,
    "LVGI": 1.037
}

# ==============================================================================
# 2. HÀM TẢI & XỬ LÝ DỮ LIỆU
# ==============================================================================
@st.cache_data
def load_and_clean_data(file_source):
    """Đọc và làm sạch dữ liệu CSV"""
    if isinstance(file_source, str):
        if not os.path.exists(file_source):
            return None, f"Không tìm thấy file: {file_source}"
        df = pd.read_csv(file_source)
    else:
        df = pd.read_csv(file_source)

    # Kiểm tra cột
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        return None, f"File thiếu các cột bắt buộc: {missing}"

    # Ép kiểu dữ liệu và lọc NA
    cleaned = df[FEATURES + [TARGET]].copy()
    for c in FEATURES + [TARGET]:
        cleaned[c] = pd.to_numeric(cleaned[c], errors="coerce")
    cleaned = cleaned.dropna().reset_index(drop=True)
    cleaned[TARGET] = cleaned[TARGET].astype(int)

    if not set(cleaned[TARGET].unique()).issubset({0, 1}):
        return None, "Cột FRAUD_FLAG chỉ được chứa giá trị 0 hoặc 1."

    return cleaned, None

@st.cache_resource
def train_model(data, test_size=0.20, random_state=42):
    """Huấn luyện Pipeline StandardScaler + Logistic Regression"""
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=5000, random_state=random_state))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline, X_train, X_test, y_train, y_test

def generate_excel_report(coef_table, cm_table, metrics_df, interpretation_df, model_info_df):
    """Tạo file Excel tổng hợp kết quả huấn luyện mô hình"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        coef_table.to_excel(writer, "Coefficients", index=False)
        cm_table.to_excel(writer, "Confusion_Matrix")
        metrics_df.to_excel(writer, "Metrics", index=False)
        interpretation_df.to_excel(writer, "Interpretation", index=False)
        model_info_df.to_excel(writer, "Model_Info", index=False)
    output.seek(0)
    return output

# ==============================================================================
# 3. THANH ĐIỀU KHIỂN BÊN TRÁI (SIDEBAR)
# ==============================================================================
st.sidebar.markdown("## ⚙️ Cấu hình Ứng dụng")

# Lựa chọn nguồn dữ liệu
data_source_opt = st.sidebar.radio(
    "Nguồn dữ liệu huấn luyện:",
    ["Dữ liệu mặc định (MScore_data.csv)", "Tải lên file CSV riêng"],
    index=0
)

data = None
err = None
default_csv_path = "MScore_data.csv"

if data_source_opt == "Dữ liệu mặc định (MScore_data.csv)":
    if os.path.exists(default_csv_path):
        data, err = load_and_clean_data(default_csv_path)
    else:
        st.sidebar.warning(f"Chưa có file '{default_csv_path}' trong thư mục. Vui lòng tải file lên.")
        uploaded_file = st.sidebar.file_uploader("Chọn file CSV:", type=["csv"])
        if uploaded_file:
            data, err = load_and_clean_data(uploaded_file)
else:
    uploaded_file = st.sidebar.file_uploader("Tải lên file CSV huấn luyện:", type=["csv"])
    if uploaded_file:
        data, err = load_and_clean_data(uploaded_file)
    else:
        st.sidebar.info("Vui lòng tải lên file CSV có 8 biến Beneish và cột FRAUD_FLAG.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Siêu tham số Mô hình")
test_size = st.sidebar.slider("Tỷ lệ tập kiểm tra (Test Size):", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
random_state = st.sidebar.number_input("Random Seed:", min_value=0, max_value=9999, value=42, step=1)
threshold = st.sidebar.slider(
    "Ngưỡng quyết định (Decision Threshold):",
    min_value=0.10, max_value=0.90, value=0.50, step=0.05,
    help="Hạ thấp ngưỡng giúp tăng độ nhạy (Recall) phát hiện gian lận nhưng có thể tăng cảnh báo nhầm (False Positive)."
)

st.sidebar.markdown("---")
st.sidebar.caption("Phát triển bởi Chuyên gia AI & Tài chính • Nền tảng Streamlit")

# ==============================================================================
# 4. GIAO DIỆN CHÍNH (MAIN INTERFACE)
# ==============================================================================
st.markdown('<div class="main-title">⚖️ Hệ Thống Dự Báo Gian Lận Báo Cáo Tài Chính</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ứng dụng Hồi quy Logistic trên nền tảng 8 chỉ số Beneish M-Score giúp kiểm toán viên và nhà đầu tư phát hiện sớm rủi ro gian lận.</div>', unsafe_allow_html=True)

if err:
    st.error(f"❌ Lỗi xử lý dữ liệu: {err}")
    st.stop()

if data is None:
    st.warning("⚠️ Chưa có dữ liệu để huấn luyện. Vui lòng tải lên file CSV ở thanh bên trái (Sidebar) để tiếp tục.")
    st.stop()

# Khởi tạo mô hình
pipeline, X_train, X_test, y_train, y_test = train_model(data, test_size=test_size, random_state=random_state)
logistic_step = pipeline.named_steps["logistic"]
intercept = logistic_step.intercept_[0]
coef = logistic_step.coef_[0]

# Dự báo tập Test
y_prob_test = pipeline.predict_proba(X_test)[:, 1]
y_pred_test = (y_prob_test >= threshold).astype(int)

# Tính toán các chỉ tiêu hiệu năng
cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
acc = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test, zero_division=0)
rec = recall_score(y_test, y_pred_test, zero_division=0)
f1 = f1_score(y_test, y_pred_test, zero_division=0)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

# Bảng hệ số & Odds Ratio
coef_df = pd.DataFrame({
    "Chỉ số": FEATURES,
    "Hệ số (Beta)": coef,
    "Odds Ratio": np.exp(coef),
    "Tác động đến gian lận": ["Tăng nguy cơ (Tích cực)" if b > 0 else "Giảm nguy cơ (Tiêu cực)" if b < 0 else "Không tác động" for b in coef]
})

# Bảng diễn giải kinh tế
interpretation_list = []
for name, b in zip(FEATURES, coef):
    or_val = np.exp(b)
    if b > 0:
        desc = f"Khi {name} tăng 1 độ lệch chuẩn, log-odds gian lận tăng {b:.4f}; tỷ số odds gian lận nhân thêm {or_val:.4f} lần (tăng nguy cơ)."
    elif b < 0:
        desc = f"Khi {name} tăng 1 độ lệch chuẩn, log-odds gian lận giảm {abs(b):.4f}; tỷ số odds gian lận nhân {or_val:.4f} lần (giảm nguy cơ)."
    else:
        desc = f"{name} không có ảnh hưởng đáng kể đến mô hình."
    interpretation_list.append({
        "Chỉ số": name,
        "Hệ số Beta": b,
        "Odds Ratio": or_val,
        "Diễn giải ý nghĩa": desc
    })
interpretation_df = pd.DataFrame(interpretation_list)

# TẠO TABS
tab_explore, tab_eval, tab_predict, tab_docs = st.tabs([
    "📊 1. Khám Phá Dữ Liệu",
    "🤖 2. Đánh Giá Mô Hình",
    "🔍 3. Dự Báo Doanh Nghiệp",
    "📖 4. Cẩm Nang & Phương Pháp"
])

# ==============================================================================
# TAB 1: KHÁM PHÁ DỮ LIỆU
# ==============================================================================
with tab_explore:
    st.subheader("📌 Tổng quan Tập Dữ liệu")
    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng số quan sát", f"{len(data):,} mẫu")
    col2.metric("Số lượng biến độc lập", f"{len(FEATURES)} chỉ số")
    fraud_count = data[TARGET].sum()
    fraud_pct = (fraud_count / len(data)) * 100
    col3.metric("Số ca gian lận (Nhãn 1)", f"{fraud_count} ({fraud_pct:.1f}%)")

    st.markdown("##### 5 dòng đầu tiên của tập dữ liệu:")
    st.dataframe(data.head(), use_container_width=True)

    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        st.markdown("##### Phân bổ biến mục tiêu (FRAUD_FLAG)")
        target_counts = data[TARGET].value_counts().rename({0: "0 - Không gian lận", 1: "1 - Gian lận"}).reset_index()
        target_counts.columns = ["Trạng thái", "Số lượng"]
        fig_pie = px.pie(
            target_counts, values="Số lượng", names="Trạng thái",
            color="Trạng thái",
            color_discrete_map={"0 - Không gian lận": "#3B82F6", "1 - Gian lận": "#EF4444"},
            hole=0.4
        )
        fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        st.markdown("##### Thống kê mô tả 8 biến Beneish")
        st.dataframe(data[FEATURES].describe().T[["mean", "std", "min", "50%", "max"]].rename(columns={"50%": "median"}), use_container_width=True)

    st.markdown("##### Ma trận tương quan (Correlation Matrix) giữa 8 biến")
    corr = data[FEATURES].corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Hệ số tương quan Pearson giữa các chỉ số Beneish"
    )
    fig_corr.update_layout(margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_corr, use_container_width=True)

# ==============================================================================
# TAB 2: ĐÁNH GIÁ MÔ HÌNH
# ==============================================================================
with tab_eval:
    st.subheader("🎯 Hiệu Năng Mô Hình Logistic Regression trên Tập Test")
    st.caption(f"Tập huấn luyện (Train): **{len(X_train)}** mẫu | Tập kiểm định (Test): **{len(X_test)}** mẫu | Ngưỡng quyết định: **{threshold:.2f}**")

    # Hiển thị các thẻ KPI
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Độ chính xác (Accuracy)", f"{acc:.2%}")
    kpi2.metric("Độ chuẩn xác (Precision)", f"{prec:.2%}", help="Trong số những ca mô hình báo gian lận, có bao nhiêu ca thực sự gian lận.")
    kpi3.metric("Độ nhạy (Recall / Sensitivity)", f"{rec:.2%}", help="Trong số những ca gian lận thực tế, mô hình bắt trúng bao nhiêu ca.")
    kpi4.metric("F1-Score", f"{f1:.2%}", help="Trung bình điều hòa giữa Precision và Recall.")

    kpi5, kpi6, kpi7 = st.columns(3)
    kpi5.metric("Độ đặc hiệu (Specificity)", f"{specificity:.2%}", help="Tỷ lệ nhận diện đúng các ca không gian lận.")
    kpi6.metric("Tỷ lệ dương tính giả (FPR)", f"{fpr:.2%}", help="Tỷ lệ báo oan doanh nghiệp bình thường thành gian lận.")
    kpi7.metric("Tỷ lệ âm tính giả (FNR)", f"{fnr:.2%}", help="Tỷ lệ bỏ sót doanh nghiệp gian lận.")

    st.markdown("---")

    col_cm, col_coef = st.columns([1, 1])

    with col_cm:
        st.markdown("##### 🔲 Ma trận nhầm lẫn (Confusion Matrix)")
        cm_data = pd.DataFrame(
            cm,
            index=["Thực tế: Không gian lận (0)", "Thực tế: Gian lận (1)"],
            columns=["Dự báo: Không gian lận (0)", "Dự báo: Gian lận (1)"]
        )
        
        # Biểu đồ heatmap ma trận nhầm lẫn
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Dự báo", y="Thực tế", color="Số lượng"),
            x=["Không gian lận (0)", "Gian lận (1)"],
            y=["Không gian lận (0)", "Gian lận (1)"],
            text_auto=True,
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

        st.info(f"""
        **Chi tiết kết quả phân loại:**
        - **TN (Đúng không gian lận):** {tn}
        - **FP (Báo oan - Dương tính giả):** {fp}
        - **FN (Bỏ sót - Âm tính giả):** {fn}
        - **TP (Bắt trúng - Dương tính thật):** {tp}
        """)

    with col_coef:
        st.markdown("##### 📈 Tác động của các biến (Hệ số hồi quy & Odds Ratio)")
        fig_coef = px.bar(
            coef_df.sort_values(by="Hệ số (Beta)", ascending=True),
            x="Hệ số (Beta)",
            y="Chỉ số",
            orientation="h",
            color="Hệ số (Beta)",
            color_continuous_scale="spectral",
            title="Trọng số hệ số Beta chuẩn hóa"
        )
        fig_coef.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=350)
        st.plotly_chart(fig_coef, use_container_width=True)

    st.markdown("##### 📝 Chi tiết Hệ số Hồi quy và Diễn giải Tác động")
    st.dataframe(interpretation_df, use_container_width=True)

    # Phương trình toán học
    equation = f"**logit(P(FRAUD=1))** = {intercept:.4f}"
    for name, b in zip(FEATURES, coef):
        sign = "+" if b >= 0 else "-"
        equation += f" {sign} {abs(b):.4f} * **{name}**_std"
    st.markdown(f"> 📐 **Phương trình hồi quy chuẩn hóa:** {equation}")

    # Xuất báo cáo Excel
    st.markdown("---")
    st.markdown("##### 📥 Xuất Toàn Bộ Kết Quả Đánh Giá ra Excel")
    
    metrics_summary_df = pd.DataFrame({
        "Chỉ tiêu": ["Accuracy", "Precision", "Recall / Sensitivity", "F1-score", "Specificity", "FPR", "FNR"],
        "Giá trị": [acc, prec, rec, f1, specificity, fpr, fnr],
        "Giá trị (%)": [acc*100, prec*100, rec*100, f1*100, specificity*100, fpr*100, fnr*100]
    })
    model_info_df = pd.DataFrame({
        "Thông số": ["Intercept", "Threshold", "Train_size", "Test_size", "Random_state"],
        "Giá trị": [intercept, threshold, len(X_train), len(X_test), random_state]
    })

    excel_data = generate_excel_report(coef_df, cm_data, metrics_summary_df, interpretation_df, model_info_df)
    st.download_button(
        label="📥 Tải Báo Cáo Excel (Logistic_Regression_MScore_Results.xlsx)",
        data=excel_data,
        file_name="Logistic_Regression_MScore_Results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ==============================================================================
# TAB 3: DỰ BÁO DOANH NGHIỆP
# ==============================================================================
with tab_predict:
    st.subheader("🔍 Công Cụ Dự Báo Nguy Cơ Gian Lận")
    
    pred_mode = st.radio("Chọn hình thức kiểm tra:", ["Kiểm tra từng doanh nghiệp (Nhập tay)", "Dự báo hàng loạt từ file CSV"], horizontal=True)

    if pred_mode == "Kiểm tra từng doanh nghiệp (Nhập tay)":
        st.markdown("##### Nhập 8 chỉ số tài chính của doanh nghiệp cần kiểm tra:")
        
        # Tiện ích nạp nhanh dữ liệu mẫu
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        if "input_vals" not in st.session_state:
            st.session_state.input_vals = {f: float(data[f].mean()) for f in FEATURES}

        if col_btn1.button("📋 Nạp mẫu Rủi ro cao (Gian lận)"):
            fraud_samples = data[data[TARGET] == 1]
            if not fraud_samples.empty:
                st.session_state.input_vals = fraud_samples.iloc[0][FEATURES].to_dict()
                st.success("Đã nạp số liệu từ doanh nghiệp có gian lận!")

        if col_btn2.button("📋 Nạp mẫu An toàn (Bình thường)"):
            normal_samples = data[data[TARGET] == 0]
            if not normal_samples.empty:
                st.session_state.input_vals = normal_samples.iloc[0][FEATURES].to_dict()
                st.success("Đã nạp số liệu từ doanh nghiệp bình thường!")

        # Form nhập liệu
        c1, c2, c3, c4 = st.columns(4)
        v_dsri = c1.number_input("DSRI (Số ngày thu tiền)", value=float(st.session_state.input_vals.get("DSRI", 1.0)), step=0.01, format="%.4f")
        v_gmi = c2.number_input("GMI (Tỷ suất LN gộp)", value=float(st.session_state.input_vals.get("GMI", 1.0)), step=0.01, format="%.4f")
        v_aqi = c3.number_input("AQI (Chất lượng tài sản)", value=float(st.session_state.input_vals.get("AQI", 1.0)), step=0.01, format="%.4f")
        v_sgi = c4.number_input("SGI (Tăng trưởng doanh thu)", value=float(st.session_state.input_vals.get("SGI", 1.0)), step=0.01, format="%.4f")

        c5, c6, c7, c8 = st.columns(4)
        v_depi = c5.number_input("DEPI (Tỷ lệ khấu hao)", value=float(st.session_state.input_vals.get("DEPI", 1.0)), step=0.01, format="%.4f")
        v_sgai = c6.number_input("SGAI (Chi phí BH & QL)", value=float(st.session_state.input_vals.get("SGAI", 1.0)), step=0.01, format="%.4f")
        v_tata = c7.number_input("TATA (Dồn tích kế toán)", value=float(st.session_state.input_vals.get("TATA", 0.05)), step=0.01, format="%.4f")
        v_lvgi = c8.number_input("LVGI (Đòn bẩy tài chính)", value=float(st.session_state.input_vals.get("LVGI", 1.0)), step=0.01, format="%.4f")

        input_df = pd.DataFrame([{
            "DSRI": v_dsri, "GMI": v_gmi, "AQI": v_aqi, "SGI": v_sgi,
            "DEPI": v_depi, "SGAI": v_sgai, "TATA": v_tata, "LVGI": v_lvgi
        }])

        if st.button("🚀 Chạy Phân Tích & Dự Báo", type="primary", use_container_width=True):
            prob = pipeline.predict_proba(input_df)[0, 1]
            is_fraud = prob >= threshold

            st.markdown("---")
            st.markdown("### Kết quả Phân Tích Rủi Ro")

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                # Đồng hồ xác suất
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Xác suất Gian lận (%)", 'font': {'size': 20}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "#EF4444" if is_fraud else "#22C55E"},
                        'steps': [
                            {'range': [0, threshold * 100], 'color': "#DCFCE7"},
                            {'range': [threshold * 100, 100], 'color': "#FEE2E2"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.8,
                            'value': threshold * 100
                        }
                    }
                ))
                fig_gauge.update_layout(height=280, margin=dict(t=40, b=20, l=30, r=30))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with res_col2:
                if is_fraud:
                    st.markdown(f"""
                    <div class="risk-high">
                        <h3>⚠️ CẢNH BÁO: NGUY CƠ GIAN LẬN CAO</h3>
                        <p>Xác suất dự báo là <b>{prob:.2%}</b>, vượt qua ngưỡng kiểm soát <b>{threshold:.2%}</b>.</p>
                        <p><b>Khuyến nghị:</b> Thực hiện kiểm toán chi tiết các khoản mục dồn tích, công nợ phải thu và chính sách ghi nhận doanh thu.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="risk-low">
                        <h3>✅ AN TOÀN: NGUY CƠ GIAN LẬN THẤP</h3>
                        <p>Xác suất dự báo là <b>{prob:.2%}</b>, nằm dưới ngưỡng rủi ro <b>{threshold:.2%}</b>.</p>
                        <p><b>Nhận định:</b> Các chỉ số tài chính nằm trong phạm vi ổn định tương đối.</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Bảng đối chiếu ngưỡng Beneish
            st.markdown("##### Đối chiếu 8 chỉ số với Ngưỡng nghi vấn kinh nghiệm của GS. Beneish:")
            check_data = []
            for col in FEATURES:
                val = input_df[col].iloc[0]
                bench = BENCHMARK_THRESHOLDS[col]
                suspicious = val > bench
                check_data.append({
                    "Chỉ số": col,
                    "Tên đầy đủ": FEATURE_DESCRIPTIONS[col],
                    "Giá trị DN": val,
                    "Ngưỡng chuẩn Beneish": f"> {bench}",
                    "Tình trạng": "🚨 Vượt ngưỡng nghi vấn" if suspicious else "Ổn định"
                })
            check_df = pd.DataFrame(check_data)
            st.dataframe(check_df, use_container_width=True)

    else:
        st.markdown("##### 📁 Tải lên file CSV chứa danh sách nhiều doanh nghiệp:")
        st.caption("File cần có đủ 8 cột: DSRI, GMI, AQI, SGI, DEPI, SGAI, TATA, LVGI (có thể kèm cột Ticker hoặc Company_Name nếu có).")
        batch_file = st.file_uploader("Chọn file CSV dự báo hàng loạt:", type=["csv"], key="batch_uploader")
        
        if batch_file:
            try:
                batch_df = pd.read_csv(batch_file)
                st.write(f"Đã đọc: **{len(batch_df)}** doanh nghiệp")
                
                missing_cols = [c for c in FEATURES if c not in batch_df.columns]
                if missing_cols:
                    st.error(f"File thiếu các cột chỉ số: {missing_cols}")
                else:
                    st.dataframe(batch_df.head(), use_container_width=True)
                    
                    if st.button("🚀 Chạy Dự Báo Cho Toàn Bộ Danh Sách", type="primary"):
                        batch_features = batch_df[FEATURES].copy()
                        for c in FEATURES:
                            batch_features[c] = pd.to_numeric(batch_features[c], errors="coerce").fillna(0)
                        
                        probs = pipeline.predict_proba(batch_features)[:, 1]
                        preds = (probs >= threshold).astype(int)
                        
                        res_batch = batch_df.copy()
                        res_batch["Xac_Suat_Gian_Lan"] = probs
                        res_batch["Xac_Suat_Gian_Lan (%)"] = (probs * 100).round(2)
                        res_batch["Du_Bao"] = preds
                        res_batch["Danh_Gia_Rui_Ro"] = [
                            "Rất cao" if p >= 0.70 else "Cao" if p >= threshold else "Trung bình" if p >= 0.30 else "Thấp"
                            for p in probs
                        ]

                        st.success("✅ Dự báo hoàn tất!")
                        
                        high_risk_count = (res_batch["Du_Bao"] == 1).sum()
                        st.warning(f"Phát hiện **{high_risk_count} / {len(res_batch)}** doanh nghiệp thuộc diện **NGUY CƠ GIAN LẬN CAO** ({high_risk_count/len(res_batch):.1%}).")
                        
                        st.dataframe(res_batch, use_container_width=True)
                        
                        # Cho phép tải về CSV
                        csv_export = res_batch.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            label="📥 Tải Kết Quả Dự Báo (CSV)",
                            data=csv_export,
                            file_name="Fraud_Prediction_Batch_Results.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"Lỗi khi xử lý file: {e}")

# ==============================================================================
# TAB 4: CẨM NANG & PHƯƠNG PHÁP
# ==============================================================================
with tab_docs:
    st.subheader("📖 Phương Pháp Luận & Cẩm Nang Beneish M-Score")
    
    st.markdown("""
    ### 1. Giới thiệu về Mô hình Beneish M-Score
    Mô hình **Beneish M-Score** được phát triển bởi Giáo sư Messod Beneish (Đại học Indiana, 1999). Đây là một mô hình toán - thống kê sử dụng **8 chỉ số tài chính** tính toán từ Báo cáo Tài chính của doanh nghiệp nhằm phát hiện khả năng ban lãnh đạo thao túng lợi nhuận (Earnings Manipulation) hoặc gian lận báo cáo tài chính.

    Trường hợp nổi tiếng nhất: Mô hình Beneish đã phát hiện chính xác dấu hiệu gian lận của tập đoàn năng lượng Enron từ trước khi vụ bê bối sụp đổ xảy ra.
    """)

    st.markdown("### 2. Ý nghĩa chi tiết của 8 chỉ số Beneish:")
    docs_table = pd.DataFrame([
        {
            "Chỉ số": "DSRI",
            "Tên tiếng Anh": "Days Sales in Receivables Index",
            "Ý nghĩa tài chính": "Tỷ lệ số ngày thu tiền khách hàng năm nay so với năm trước. Nếu DSRI tăng đột biến (> 1), doanh thu có thể được ghi nhận vội vàng hoặc doanh thu ảo chưa thu được tiền."
        },
        {
            "Chỉ số": "GMI",
            "Tên tiếng Anh": "Gross Margin Index",
            "Ý nghĩa tài chính": "Tỷ số giữa biên lợi nhuận gộp năm trước so với năm nay. Nếu GMI > 1, biên lợi nhuận đang suy giảm, tạo áp lực cho ban điều hành phải 'thổi phồng' lợi nhuận để làm đẹp BCTC."
        },
        {
            "Chỉ số": "AQI",
            "Tên tiếng Anh": "Asset Quality Index",
            "Ý nghĩa tài chính": "Tỷ số chất lượng tài sản (tài sản phi vãng hạn trừ TSCĐ thuần). AQI > 1 cho thấy doanh nghiệp đang vốn hóa chi phí vào tài sản thay vì ghi nhận vào chi phí kỳ này."
        },
        {
            "Chỉ số": "SGI",
            "Tên tiếng Anh": "Sales Growth Index",
            "Ý nghĩa tài chính": "Tỷ lệ tăng trưởng doanh thu. Tăng trưởng nóng tạo áp lực duy trì kỳ vọng thị trường, dễ dẫn đến hành vi gian lận ghi nhận doanh thu sớm."
        },
        {
            "Chỉ số": "DEPI",
            "Tên tiếng Anh": "Depreciation Index",
            "Ý nghĩa tài chính": "Tỷ số tỷ lệ khấu hao năm trước so với năm nay. DEPI > 1 cho thấy tốc độ trích khấu hao đang chậm lại, doanh nghiệp có thể đã kéo dài thời gian khấu hao để giảm chi phí."
        },
        {
            "Chỉ số": "SGAI",
            "Tên tiếng Anh": "Sales, General & Administrative Expenses Index",
            "Ý nghĩa tài chính": "Tỷ lệ chi phí bán hàng và quản lý so với doanh thu. Nếu SGAI > 1, hiệu quả hoạt động đang giảm sút."
        },
        {
            "Chỉ số": "TATA",
            "Tên tiếng Anh": "Total Accruals to Total Assets",
            "Ý nghĩa tài chính": "Biến kế toán dồn tích (Lợi nhuận thuần trừ dòng tiền hoạt động) chia cho Tổng tài sản. TATA càng cao chứng tỏ lợi nhuận chủ yếu nằm trên giấy chứ không thu được tiền mặt."
        },
        {
            "Chỉ số": "LVGI",
            "Tên tiếng Anh": "Leverage Index",
            "Ý nghĩa tài chính": "Tỷ lệ nợ trên tổng tài sản năm nay so với năm trước. Đòn bẩy tài chính tăng cao làm gia tăng áp lực trả nợ và động cơ làm giả số liệu để đáp ứng các cam kết tín dụng."
        }
    ])
    st.table(docs_table)

    st.markdown("""
    ### 3. Mô hình Hồi quy Logistic (Logistic Regression)
    Khác với công thức trọng số cố định ban đầu của Beneish (năm 1999 tại thị trường Mỹ), ứng dụng này sử dụng **Hồi quy Logistic được huấn luyện trên dữ liệu thực nghiệm**:
    - **Chuẩn hóa (StandardScaler):** Đưa các chỉ số về cùng thang đo (trung bình = 0, độ lệch chuẩn = 1) nhằm đảm bảo tính khách quan giữa các biến có biên độ dao động khác nhau.
    - **Ước lượng tham số (Maximum Likelihood Estimation):** Tìm hệ số Beta tối ưu nhất để phân loại giữa doanh nghiệp Gian lận (Nhãn 1) và Bình thường (Nhãn 0).
    - **Tỷ số Odds Ratio:** Đo lường mức độ thay đổi xác suất khi một chỉ số thay đổi 1 đơn vị độ lệch chuẩn.
    """)
