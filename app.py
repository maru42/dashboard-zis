import streamlit as st
import pandas as pd
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Dashboard ZIS | Overview",
    page_icon="📊",
    layout="wide"
)

# --- CUSTOM CSS (Diterapkan ke semua halaman) ---
st.markdown("""
<style>
/* General Body Styles */
body {
    background-color: #f0f2f6;
}

/* Main container styling */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* Card Styling */
.card {
    background-color: white;
    border-radius: 12px;
    padding: 25px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease-in-out;
    height: 100%;
}
.card:hover {
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
}

/* Metric Card Styling */
.metric-card { text-align: center; }
.metric-card h3 { font-size: 1.2rem; color: #6c757d; margin-bottom: 0.5rem; }
.metric-card p { font-size: 2.5rem; font-weight: 600; color: #1E293B; margin-bottom: 0.5rem; }
.metric-card span { font-size: 1rem; color: #28a745; }
.metric-card span.red { color: #dc3545; }

/* Button Styling */
div.stButton > button {
    background-color: #0068c9;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
    font-weight: 500;
    width: 100%;
}
div.stButton > button:hover { background-color: #0055a8; }

/* Header styling */
h1, h2, h3, h4 { color: #1E293B; }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'scaled_data' not in st.session_state:
    st.session_state.scaled_data = None
if 'result_df' not in st.session_state:
    st.session_state.result_df = None
if 'model' not in st.session_state:
    st.session_state.model = None

# --- SIDEBAR FOR UPLOAD ---
with st.sidebar:
    st.title("Upload Data Anda")
    uploaded_file = st.file_uploader("Upload file Rekapitulasi ZIS", type=["xlsx", "csv"])
    
    if uploaded_file:
        if st.session_state.df is None or not st.session_state.df.equals(pd.read_excel(uploaded_file)):
            st.session_state.df = pd.read_excel(uploaded_file)
            st.session_state.df_processed = None
            st.session_state.scaled_data = None
            st.session_state.result_df = None
            st.session_state.model = None
            st.success("File berhasil diunggah! Silakan mulai analisis dari halaman Preprocessing.")

# --- HOMEPAGE CONTENT ---
st.title("📊 Overview Dashboard ZIS")
st.markdown("Selamat datang! Halaman ini menampilkan ringkasan data awal dari file yang Anda unggah.")

if st.session_state.df is not None:
    
    st.markdown("### Ringkasan Data ZIS")
    
    df_display = st.session_state.df.copy()
    
    # --- Calculate Metrics ---
    zakat_beras_col = "Jumlah Beras (Kg)"
    
    # 1. Calculate total rice zakat
    total_zakat_beras = 0
    if zakat_beras_col in df_display.columns:
        numeric_beras = pd.to_numeric(df_display[zakat_beras_col], errors='coerce')
        total_zakat_beras = numeric_beras.sum()

    # 2. Calculate total money donation (excluding rice column)
    numeric_cols = df_display.select_dtypes(include=np.number)
    if zakat_beras_col in numeric_cols.columns:
        numeric_cols_money = numeric_cols.drop(columns=[zakat_beras_col])
    else:
        numeric_cols_money = numeric_cols
    total_donasi_uang = numeric_cols_money.sum().sum()

    # 3. Other metrics
    jumlah_transaksi = len(df_display)
    rata_donasi = numeric_cols_money.sum(axis=1).mean() if not numeric_cols_money.empty else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="card metric-card">
            <h3>Total Zakat Uang</h3><p>Rp {total_donasi_uang:,.0f}</p><span>Terkumpul</span>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="card metric-card">
            <h3>Jumlah Transaksi</h3><p>{jumlah_transaksi}</p><span>Tercatat</span>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="card metric-card">
            <h3>Total Zakat Beras</h3><p>{total_zakat_beras:,.1f}</p><span>Kg</span>
        </div>""", unsafe_allow_html=True)
    with col4:
         st.markdown(f"""
        <div class="card metric-card">
            <h3>Rata-Rata Zakat Uang</h3><p>Rp {rata_donasi:,.0f}</p><span>per Transaksi</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Pratinjau Data Awal")
    st.dataframe(st.session_state.df.head())
    
else:
    st.info("Silakan unggah file Excel di sidebar untuk memulai analisis.")