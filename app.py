# -*- coding: utf-8 -*-
"""
Web App: Dự báo Gian lận Báo cáo Tài chính (Financial Statement Fraud Detection)
Mô hình tích hợp: Hồi quy Logistic (Logistic Regression) & XGBoost Classifier
Dữ liệu: 8 biến chỉ số Beneish M-Score
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

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ==============================================================================
# 1. CẤU HÌNH TRANG WEB & THEME
# ==============================================================================
st.set_page_config(
    page_title="Dự Báo Gian Lận BCTC | Logistic Regression & XGBoost",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tùy biến CSS tăng tính thẩm mỹ & chuyên nghiệp
st.markdown("""
<style>
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
    .risk-high {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
        padding: 15px;
        border-radius: 8px;
        color: #991B1B;
    }
    .risk-medium {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 15px;
        border-radius: 8px;
        color: #92400E;
    }
    .risk-low {
        background-color: #DCFCE7;
        border-left: 5px solid #22C55E;
        padding: 15px;
        border-radius: 8px;
        color: #166534;
    }
    .best-badge {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
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
# 2. HÀM TẢI & XỬ LÝ DỮ LIỆU VÀ MÔ HÌNH
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
def train_models(data, test_size=0.20, random_state=42, xgb_depth=4, xgb_trees=100, xgb_lr=0.05):
    """Huấn luyện đồng thời cả Logistic Regression và XGBoost"""
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 1. Huấn luyện Logistic Regression
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=5000, random_state=random_state))
    ])
    lr_pipeline.fit(X_train, y_train)

    # 2. Huấn luyện XGBoost
    xgb_pipeline = None
    if XGB_AVAILABLE:
        xgb_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", XGBClassifier(
                n_estimators=xgb_trees,
                max_depth=xgb_depth,
                learning_rate=xgb_lr,
                random_state=random_state,
                eval_metric="logloss"
            ))
        ])
        xgb_pipeline.fit(X_train, y_train)

    return (lr_pipeline, xgb_pipeline), X_train, X_test, y_train, y_test

def compute_metrics(y_true, y_pred, y_prob):
    """Tính toán chi tiết các chỉ số hiệu năng và ma trận nhầm lẫn"""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "cm": cm, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "spec": spec, "fpr": fpr, "fnr": fnr
    }

def generate_full_excel_report(lr_coef_df, lr_cm_df, lr_metrics_df, xgb_cm_df, xgb_metrics_df, xgb_importance_df, comparison_df, interpretation_df, model_info_df):
    """Tạo file Excel tổng hợp kết quả của cả hai mô hình"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        comparison_df.to_excel(writer, "Model_Comparison", index=False)
        lr_metrics_df.to_excel(writer, "Logistic_Metrics", index=False)
        lr_coef_df.to_excel(writer, "Logistic_Coefficients", index=False)
        lr_cm_df.to_excel(writer, "Logistic_Confusion_Matrix")
        if xgb_metrics_df is not None:
            xgb_metrics_df.to_excel(writer, "XGBoost_Metrics", index=False)
        if xgb_importance_df is not None:
            xgb_importance_df.to_excel(writer, "XGBoost_Importance", index=False)
        if xgb_cm_df is not None:
            xgb_cm_df.to_excel(writer, "XGBoost_Confusion_Matrix")
        interpretation_df.to_excel(writer, "Logistic_Interpretation", index=False)
        model_info_df.to_excel(writer, "Model_Info", index=False)
    output.seek(0)
    return output

# ==============================================================================
# 3. THANH ĐIỀU KHIỂN BÊN TRÁI (SIDEBAR)
# ==============================================================================
st.sidebar.markdown("## ⚙️ Cấu hình Hệ thống")

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
st.sidebar.markdown("### 🎛️ Tham Số Chung")
test_size = st.sidebar.slider("Tỷ lệ tập kiểm tra (Test Size):", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
random_state = st.sidebar.number_input("Random Seed:", min_value=0, max_value=9999, value=42, step=1)
threshold = st.sidebar.slider(
    "Ngưỡng quyết định (Decision Threshold):",
    min_value=0.10, max_value=0.90, value=0.50, step=0.05,
    help="Hạ thấp ngưỡng giúp tăng độ nhạy (Recall) bắt gian lận, tăng ngưỡng giúp giảm báo động sai (Precision)."
)

# Cấu hình XGBoost
st.sidebar.markdown("---")
with st.sidebar.expander("🌳 Siêu tham số XGBoost", expanded=False):
    xgb_trees = st.slider("Số lượng cây (n_estimators):", 50, 300, 100, step=25)
    xgb_depth = st.slider("Độ sâu tối đa cây (max_depth):", 2, 10, 4, step=1)
    xgb_lr = st.select_slider("Tốc độ học (learning_rate):", options=[0.01, 0.03, 0.05, 0.1, 0.2], value=0.05)

st.sidebar.markdown("---")
st.sidebar.caption("⚖️ Beneish M-Score Fraud Detector • Streamlit App")

# ==============================================================================
# 4. XỬ LÝ CHÍNH & HUẤN LUYỆN
# ==============================================================================
st.markdown('<div class="main-title">⚖️ Hệ Thống Dự Báo Gian Lận Báo Cáo Tài Chính</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Tích hợp mô hình <b>Hồi quy Logistic</b> và <b>XGBoost Classifier</b> trên nền tảng 8 chỉ số <b>Beneish M-Score</b>.</div>', unsafe_allow_html=True)

if not XGB_AVAILABLE:
    st.warning("⚠️ Thư viện `xgboost` chưa được cài đặt. Ứng dụng sẽ hoạt động ở chế độ Hồi quy Logistic. Hãy chạy `pip install xgboost` để kích hoạt đầy đủ.")

if err:
    st.error(f"❌ Lỗi xử lý dữ liệu: {err}")
    st.stop()

if data is None:
    st.warning("⚠️ Chưa có dữ liệu để huấn luyện. Vui lòng chọn hoặc tải lên file CSV ở thanh bên trái (Sidebar).")
    st.stop()

# Huấn luyện mô hình
(lr_pipeline, xgb_pipeline), X_train, X_test, y_train, y_test = train_models(
    data, test_size=test_size, random_state=random_state,
    xgb_depth=xgb_depth, xgb_trees=xgb_trees, xgb_lr=xgb_lr
)

# Dự báo Logistic Regression
lr_prob_test = lr_pipeline.predict_proba(X_test)[:, 1]
lr_pred_test = (lr_prob_test >= threshold).astype(int)
lr_res = compute_metrics(y_test, lr_pred_test, lr_prob_test)

# Lấy hệ số Logistic
logistic_step = lr_pipeline.named_steps["logistic"]
intercept = logistic_step.intercept_[0]
coef = logistic_step.coef_[0]

lr_coef_df = pd.DataFrame({
    "Chỉ số": FEATURES,
    "Hệ số (Beta)": coef,
    "Odds Ratio": np.exp(coef),
    "Tác động đến gian lận": ["Tăng nguy cơ (Tích cực)" if b > 0 else "Giảm nguy cơ (Tiêu cực)" if b < 0 else "Không tác động" for b in coef]
})

# Diễn giải kinh tế
interpretation_list = []
for name, b in zip(FEATURES, coef):
    or_val = np.exp(b)
    if b > 0:
        desc = f"Khi {name} tăng 1 độ lệch chuẩn, log-odds gian lận tăng {b:.4f}; odds gian lận nhân thêm {or_val:.4f} lần (tăng nguy cơ)."
    elif b < 0:
        desc = f"Khi {name} tăng 1 độ lệch chuẩn, log-odds gian lận giảm {abs(b):.4f}; odds gian lận nhân {or_val:.4f} lần (giảm nguy cơ)."
    else:
        desc = f"{name} không có ảnh hưởng đáng kể đến mô hình."
    interpretation_list.append({
        "Chỉ số": name,
        "Hệ số Beta": b,
        "Odds Ratio": or_val,
        "Diễn giải ý nghĩa": desc
    })
interpretation_df = pd.DataFrame(interpretation_list)

# Dự báo XGBoost (nếu có)
xgb_res = None
xgb_importance_df = None
if xgb_pipeline is not None:
    xgb_prob_test = xgb_pipeline.predict_proba(X_test)[:, 1]
    xgb_pred_test = (xgb_prob_test >= threshold).astype(int)
    xgb_res = compute_metrics(y_test, xgb_pred_test, xgb_prob_test)

    # Feature Importance của XGBoost
    xgb_step = xgb_pipeline.named_steps["xgb"]
    importances = xgb_step.feature_importances_
    xgb_importance_df = pd.DataFrame({
        "Chỉ số": FEATURES,
        "Độ quan trọng (Importance)": importances
    }).sort_values(by="Độ quan trọng (Importance)", ascending=False)

# Bảng so sánh 2 mô hình
comp_metrics = [
    ("Độ chính xác (Accuracy)", lr_res["acc"], xgb_res["acc"] if xgb_res else np.nan, "higher"),
    ("Độ chuẩn xác (Precision)", lr_res["prec"], xgb_res["prec"] if xgb_res else np.nan, "higher"),
    ("Độ nhạy (Recall / Sensitivity)", lr_res["rec"], xgb_res["rec"] if xgb_res else np.nan, "higher"),
    ("F1-Score", lr_res["f1"], xgb_res["f1"] if xgb_res else np.nan, "higher"),
    ("Độ đặc hiệu (Specificity)", lr_res["spec"], xgb_res["spec"] if xgb_res else np.nan, "higher"),
    ("Tỷ lệ dương tính giả (FPR)", lr_res["fpr"], xgb_res["fpr"] if xgb_res else np.nan, "lower"),
    ("Tỷ lệ âm tính giả (FNR)", lr_res["fnr"], xgb_res["fnr"] if xgb_res else np.nan, "lower"),
]

comparison_rows = []
for label, val_lr, val_xgb, opt in comp_metrics:
    if np.isnan(val_xgb):
        winner = "Logistic Regression"
    elif opt == "higher":
        winner = "XGBoost" if val_xgb > val_lr else ("Logistic Regression" if val_lr > val_xgb else "Tương đương")
    else:
        winner = "XGBoost" if val_xgb < val_lr else ("Logistic Regression" if val_lr < val_xgb else "Tương đương")
    
    comparison_rows.append({
        "Chỉ tiêu đánh giá": label,
        "Logistic Regression": f"{val_lr:.2%}",
        "XGBoost": f"{val_xgb:.2%}" if not np.isnan(val_xgb) else "N/A",
        "Mô hình vượt trội": winner
    })
comparison_df = pd.DataFrame(comparison_rows)

# TẠO TABS
tab_explore, tab_eval, tab_predict, tab_docs = st.tabs([
    "📊 1. Khám Phá Dữ Liệu",
    "🤖 2. Đánh Giá & So Sánh Mô Hình",
    "🔍 3. Dự Báo Doanh Nghiệp (Đa Mô Hình)",
    "📖 4. Cẩm Nang & Phương Pháp"
])

# ==============================================================================
# TAB 1: KHÁM PHÁ DỮ LIỆU
# ==============================================================================
with tab_explore:
    st.subheader("📌 Tổng quan Tập Dữ liệu Huấn Luyện")
    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng số quan sát", f"{len(data):,} mẫu")
    col2.metric("Số lượng biến độc lập", f"{len(FEATURES)} chỉ số Beneish")
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
# TAB 2: ĐÁNH GIÁ & SO SÁNH MÔ HÌNH
# ==============================================================================
with tab_eval:
    st.subheader("⚔️ So Sánh Hiệu Năng: Logistic Regression vs. XGBoost")
    st.caption(f"Tập Train: **{len(X_train)}** mẫu | Tập Test: **{len(X_test)}** mẫu | Ngưỡng quyết định: **{threshold:.2f}**")

    # Bảng so sánh trực quan
    st.dataframe(comparison_df, use_container_width=True)

    # Hiển thị song song Ma trận nhầm lẫn
    st.markdown("---")
    st.markdown("### 🔲 So sánh Ma trận nhầm lẫn (Confusion Matrix)")
    col_cm_lr, col_cm_xgb = st.columns(2)

    with col_cm_lr:
        st.markdown("#### 1. Hồi quy Logistic")
        fig_cm_lr = px.imshow(
            lr_res["cm"],
            labels=dict(x="Dự báo", y="Thực tế", color="Số lượng"),
            x=["Không gian lận (0)", "Gian lận (1)"],
            y=["Không gian lận (0)", "Gian lận (1)"],
            text_auto=True,
            color_continuous_scale="Blues"
        )
        fig_cm_lr.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
        st.plotly_chart(fig_cm_lr, use_container_width=True)
        st.markdown(f"""
        - **Bắt trúng gian lận (TP):** `{lr_res['tp']}` | **Bỏ sót (FN):** `{lr_res['fn']}`
        - **Bắt đúng bình thường (TN):** `{lr_res['tn']}` | **Báo oan (FP):** `{lr_res['fp']}`
        - **Recall (Độ nhạy):** `{lr_res['rec']:.2%}` | **F1-Score:** `{lr_res['f1']:.2%}`
        """)

    with col_cm_xgb:
        st.markdown("#### 2. XGBoost Classifier")
        if xgb_res is not None:
            fig_cm_xgb = px.imshow(
                xgb_res["cm"],
                labels=dict(x="Dự báo", y="Thực tế", color="Số lượng"),
                x=["Không gian lận (0)", "Gian lận (1)"],
                y=["Không gian lận (0)", "Gian lận (1)"],
                text_auto=True,
                color_continuous_scale="Greens"
            )
            fig_cm_xgb.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
            st.plotly_chart(fig_cm_xgb, use_container_width=True)
            st.markdown(f"""
            - **Bắt trúng gian lận (TP):** `{xgb_res['tp']}` | **Bỏ sót (FN):** `{xgb_res['fn']}`
            - **Bắt đúng bình thường (TN):** `{xgb_res['tn']}` | **Báo oan (FP):** `{xgb_res['fp']}`
            - **Recall (Độ nhạy):** `{xgb_res['rec']:.2%}` | **F1-Score:** `{xgb_res['f1']:.2%}`
            """)
        else:
            st.info("XGBoost chưa khả dụng.")

    st.markdown("---")
    st.markdown("### 📈 Phân Tích Độ Quan Trọng và Trọng Số Của 8 Biến")
    col_w_lr, col_w_xgb = st.columns(2)

    with col_w_lr:
        st.markdown("##### Trọng số Hồi quy Logistic (Beta & Odds Ratio)")
        fig_coef = px.bar(
            lr_coef_df.sort_values(by="Hệ số (Beta)", ascending=True),
            x="Hệ số (Beta)",
            y="Chỉ số",
            orientation="h",
            color="Hệ số (Beta)",
            color_continuous_scale="spectral",
            title="Hệ số Beta chuẩn hóa (Logistic Regression)"
        )
        fig_coef.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=330)
        st.plotly_chart(fig_coef, use_container_width=True)

    with col_w_xgb:
        st.markdown("##### Độ quan trọng của các chỉ số trong XGBoost (Feature Importance)")
        if xgb_importance_df is not None:
            fig_imp = px.bar(
                xgb_importance_df.sort_values(by="Độ quan trọng (Importance)", ascending=True),
                x="Độ quan trọng (Importance)",
                y="Chỉ số",
                orientation="h",
                color="Độ quan trọng (Importance)",
                color_continuous_scale="Teal",
                title="Tầm quan trọng đặc trưng (XGBoost Feature Importance)"
            )
            fig_imp.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=330)
            st.plotly_chart(fig_imp, use_container_width=True)

    # Diễn giải chi tiết
    st.markdown("##### 📝 Chi tiết Hệ số Logistic và Diễn giải Tác động Kinh tế")
    st.dataframe(interpretation_df, use_container_width=True)

    # Xuất báo cáo Excel
    st.markdown("---")
    st.markdown("##### 📥 Xuất Báo Cáo So Sánh Toàn Bộ Kết Quả Ra Excel")
    
    lr_cm_data = pd.DataFrame(
        lr_res["cm"],
        index=["Thực tế: Không gian lận (0)", "Thực tế: Gian lận (1)"],
        columns=["Dự báo: Không gian lận (0)", "Dự báo: Gian lận (1)"]
    )
    lr_metrics_summary = pd.DataFrame({
        "Chỉ tiêu": ["Accuracy", "Precision", "Recall / Sensitivity", "F1-score", "Specificity", "FPR", "FNR"],
        "Giá trị": [lr_res["acc"], lr_res["prec"], lr_res["rec"], lr_res["f1"], lr_res["spec"], lr_res["fpr"], lr_res["fnr"]],
        "Giá trị (%)": [lr_res["acc"]*100, lr_res["prec"]*100, lr_res["rec"]*100, lr_res["f1"]*100, lr_res["spec"]*100, lr_res["fpr"]*100, lr_res["fnr"]*100]
    })

    xgb_cm_data = None
    xgb_metrics_summary = None
    if xgb_res is not None:
        xgb_cm_data = pd.DataFrame(
            xgb_res["cm"],
            index=["Thực tế: Không gian lận (0)", "Thực tế: Gian lận (1)"],
            columns=["Dự báo: Không gian lận (0)", "Dự báo: Gian lận (1)"]
        )
        xgb_metrics_summary = pd.DataFrame({
            "Chỉ tiêu": ["Accuracy", "Precision", "Recall / Sensitivity", "F1-score", "Specificity", "FPR", "FNR"],
            "Giá trị": [xgb_res["acc"], xgb_res["prec"], xgb_res["rec"], xgb_res["f1"], xgb_res["spec"], xgb_res["fpr"], xgb_res["fnr"]],
            "Giá trị (%)": [xgb_res["acc"]*100, xgb_res["prec"]*100, xgb_res["rec"]*100, xgb_res["f1"]*100, xgb_res["spec"]*100, xgb_res["fpr"]*100, xgb_res["fnr"]*100]
        })

    model_info_df = pd.DataFrame({
        "Thông số": ["Threshold", "Train_size", "Test_size", "Random_state", "XGB_Trees", "XGB_Depth", "XGB_LR"],
        "Giá trị": [threshold, len(X_train), len(X_test), random_state, xgb_trees, xgb_depth, xgb_lr]
    })

    excel_data = generate_full_excel_report(
        lr_coef_df, lr_cm_data, lr_metrics_summary,
        xgb_cm_data, xgb_metrics_summary, xgb_importance_df,
        comparison_df, interpretation_df, model_info_df
    )
    st.download_button(
        label="📥 Tải Báo Cáo Excel Tổng Hợp (Logistic_Regression_XGBoost_Results.xlsx)",
        data=excel_data,
        file_name="Logistic_Regression_XGBoost_Results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ==============================================================================
# TAB 3: DỰ BÁO DOANH NGHIỆP
# ==============================================================================
with tab_predict:
    st.subheader("🔍 Công Cụ Dự Báo Nguy Cơ Gian Lận Đa Mô Hình")
    
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

        if st.button("🚀 Chạy Phân Tích & Dự Báo (Cả 2 Mô Hình)", type="primary", use_container_width=True):
            lr_p = lr_pipeline.predict_proba(input_df)[0, 1]
            lr_is_fraud = lr_p >= threshold

            xgb_p = xgb_pipeline.predict_proba(input_df)[0, 1] if xgb_pipeline else None
            xgb_is_fraud = (xgb_p >= threshold) if xgb_p is not None else False

            st.markdown("---")
            st.markdown("### Kết Quả Đánh Giá Rủi Ro Đa Mô Hình")

            col_g1, col_g2 = st.columns(2)

            with col_g1:
                st.markdown("#### 1. Hồi quy Logistic")
                fig_g_lr = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=lr_p * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Xác suất Gian lận (%)", 'font': {'size': 18}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#EF4444" if lr_is_fraud else "#22C55E"},
                        'steps': [
                            {'range': [0, threshold * 100], 'color': "#DCFCE7"},
                            {'range': [threshold * 100, 100], 'color': "#FEE2E2"}
                        ],
                        'threshold': {'line': {'color': "black", 'width': 3}, 'value': threshold * 100}
                    }
                ))
                fig_g_lr.update_layout(height=240, margin=dict(t=30, b=20, l=30, r=30))
                st.plotly_chart(fig_g_lr, use_container_width=True)

                if lr_is_fraud:
                    st.error(f"⚠️ **Logistic:** CẢNH BÁO GIAN LẬN ({lr_p:.2%})")
                else:
                    st.success(f"✅ **Logistic:** An toàn ({lr_p:.2%})")

            with col_g2:
                st.markdown("#### 2. XGBoost Classifier")
                if xgb_p is not None:
                    fig_g_xgb = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=xgb_p * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Xác suất Gian lận (%)", 'font': {'size': 18}},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#EF4444" if xgb_is_fraud else "#22C55E"},
                            'steps': [
                                {'range': [0, threshold * 100], 'color': "#DCFCE7"},
                                {'range': [threshold * 100, 100], 'color': "#FEE2E2"}
                            ],
                            'threshold': {'line': {'color': "black", 'width': 3}, 'value': threshold * 100}
                        }
                    ))
                    fig_g_xgb.update_layout(height=240, margin=dict(t=30, b=20, l=30, r=30))
                    st.plotly_chart(fig_g_xgb, use_container_width=True)

                    if xgb_is_fraud:
                        st.error(f"⚠️ **XGBoost:** CẢNH BÁO GIAN LẬN ({xgb_p:.2%})")
                    else:
                        st.success(f"✅ **XGBoost:** An toàn ({xgb_p:.2%})")
                else:
                    st.info("XGBoost chưa khả dụng.")

            # Đánh giá đồng thuận
            st.markdown("#### 🎯 Nhận Định Tổng Hợp & Khuyến Nghị Kiểm Toán")
            if lr_is_fraud and xgb_is_fraud:
                st.markdown(f"""
                <div class="risk-high">
                    <h4>🚨 CẢNH BÁO CAO ĐỘ: ĐỒNG THUẬN GIAN LẬN TỪ CẢ HAI MÔ HÌNH</h4>
                    <p>Cả <b>Logistic Regression ({lr_p:.2%})</b> và <b>XGBoost ({xgb_p:.2%})</b> đều xác định doanh nghiệp vượt ngưỡng kiểm soát rủi ro ({threshold:.2%}).</p>
                    <p><b>Khuyến nghị hành động:</b> Kiểm tra chuyên sâu chứng từ bán hàng dồn tích, chính sách ghi nhận doanh thu và khoản phải thu ngắn hạn.</p>
                </div>
                """, unsafe_allow_html=True)
            elif lr_is_fraud or xgb_is_fraud:
                suspect_model = "Logistic Regression" if lr_is_fraud else "XGBoost"
                safe_model = "XGBoost" if lr_is_fraud else "Logistic Regression"
                st.markdown(f"""
                <div class="risk-medium">
                    <h4>⚡ CẢNH BÁO PHÂN KỲ: CÓ DẤU HIỆU BẤT THƯỜNG</h4>
                    <p>Mô hình <b>{suspect_model}</b> cảnh báo nguy cơ, trong khi <b>{safe_model}</b> đánh giá an toàn. Hiện tượng này thường xuất hiện khi doanh nghiệp có một vài chỉ số tăng đột biến phi tuyến tính.</p>
                    <p><b>Khuyến nghị:</b> Rà soát thêm các chỉ số vượt ngưỡng nghi vấn bên dưới.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h4>✅ AN TOÀN ĐỒNG THUẬN: RỦI RO GIAN LẬN RẤT THẤP</h4>
                    <p>Cả hai mô hình đều đồng thuận xác suất rủi ro nằm dưới ngưỡng cảnh báo. Báo cáo tài chính phản ánh cấu trúc tương đối trung thực.</p>
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
        st.caption("File cần có đủ 8 cột: DSRI, GMI, AQI, SGI, DEPI, SGAI, TATA, LVGI (có thể kèm cột Ticker hoặc Company_Name).")
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
                    
                    if st.button("🚀 Chạy Dự Báo Đồng Thời Bằng Cả Hai Mô Hình", type="primary"):
                        batch_features = batch_df[FEATURES].copy()
                        for c in FEATURES:
                            batch_features[c] = pd.to_numeric(batch_features[c], errors="coerce").fillna(0)
                        
                        # Dự báo Logistic
                        lr_probs = lr_pipeline.predict_proba(batch_features)[:, 1]
                        lr_preds = (lr_probs >= threshold).astype(int)

                        # Dự báo XGBoost
                        xgb_probs = xgb_pipeline.predict_proba(batch_features)[:, 1] if xgb_pipeline else np.zeros(len(batch_features))
                        xgb_preds = (xgb_probs >= threshold).astype(int) if xgb_pipeline else np.zeros(len(batch_features), dtype=int)
                        
                        res_batch = batch_df.copy()
                        res_batch["Xac_Suat_Logistic (%)"] = (lr_probs * 100).round(2)
                        res_batch["Du_Bao_Logistic"] = lr_preds
                        res_batch["Xac_Suat_XGBoost (%)"] = (xgb_probs * 100).round(2)
                        res_batch["Du_Bao_XGBoost"] = xgb_preds

                        # Đánh giá đồng thuận
                        consensus_list = []
                        for lp, xp in zip(lr_preds, xgb_preds):
                            if lp == 1 and xp == 1:
                                consensus_list.append("Rủi ro rất cao (Cả 2 cảnh báo)")
                            elif lp == 1 or xp == 1:
                                consensus_list.append("Nghi vấn (1 mô hình cảnh báo)")
                            else:
                                consensus_list.append("An toàn (Cả 2 đồng thuận)")
                        res_batch["Danh_Gia_Tong_Hop"] = consensus_list

                        st.success("✅ Dự báo hoàn tất!")

                        col_sum1, col_sum2 = st.columns(2)
                        both_fraud = (res_batch["Danh_Gia_Tong_Hop"] == "Rủi ro rất cao (Cả 2 cảnh báo)").sum()
                        any_fraud = (res_batch["Danh_Gia_Tong_Hop"] != "An toàn (Cả 2 đồng thuận)").sum()

                        col_sum1.metric("Doanh nghiệp nguy cơ rất cao (Đồng thuận)", f"{both_fraud} DN ({both_fraud/len(res_batch):.1%})")
                        col_sum2.metric("Doanh nghiệp có dấu hiệu nghi vấn", f"{any_fraud} DN ({any_fraud/len(res_batch):.1%})")

                        st.dataframe(res_batch, use_container_width=True)
                        
                        # Xuất file kết quả
                        csv_export = res_batch.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            label="📥 Tải Toàn Bộ Kết Quả Dự Báo Đa Mô Hình (CSV)",
                            data=csv_export,
                            file_name="Fraud_Prediction_MultiModel_Results.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"Lỗi khi xử lý file: {e}")

# ==============================================================================
# TAB 4: CẨM NANG & PHƯƠNG PHÁP
# ==============================================================================
with tab_docs:
    st.subheader("📖 Phương Pháp Luận & Cơ Sở Lý Thuyết")
    
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
    ### 3. So sánh Hai Thuật Toán: Hồi Quy Logistic vs. XGBoost

    | Tiêu chí | Hồi quy Logistic (Logistic Regression) | XGBoost Classifier (Extreme Gradient Boosting) |
    | :--- | :--- | :--- |
    | **Bản chất toán học** | Mô hình tuyến tính tổng quát (Generalized Linear Model), phân tách không gian bằng siêu phẳng | Tập hợp cây quyết định (Ensemble Trees) tối ưu hóa bằng Gradient Boosting |
    | **Khả năng bắt phi tuyến** | Hạn chế (chủ yếu bắt mối quan hệ đơn điệu tuyến tính của log-odds) | **Xuất sắc**: Tự động nhận diện các mối tương tác phi tuyến phức tạp giữa các chỉ số |
    | **Tính giải thích (Interpretability)** | **Rất cao**: Thể hiện trực tiếp qua hệ số $\\beta$ và Tỷ số Odds Ratio | Vừa phải: Thể hiện qua tầm quan trọng đặc trưng (Feature Importance) |
    | **Tính ổn định & Kháng nhiễu** | Ổn định cao với tập dữ liệu nhỏ | Cần tinh chỉnh siêu tham số cẩn thận để tránh overfitting |
    | **Ứng dụng kiểm toán thực tế** | Dễ bảo vệ và giải trình trước Hội đồng kiểm toán hoặc Tòa án | Rất mạnh mẽ trong phát hiện các thủ thuật gian lận tinh vi được che giấu chéo |

    > 💡 **Khuyến nghị thực tiễn:** Hãy sử dụng kết quả **đồng thuận** của cả hai mô hình. Nếu cả hai đều báo nguy cơ cao, xác suất xảy ra gian lận thực tế là cực kỳ lớn!
    """)
