import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Preprocessing Data", layout="wide")

st.title("Preprocessing & Cleaning Data")
st.markdown("Lakukan pembersihan, penanganan outlier, dan normalisasi data Anda di sini.")

if st.session_state.df is None:
    st.warning("Data belum diunggah. Silakan kembali ke halaman utama untuk mengunggah file Anda.")
    st.stop()

# --- STEP 1: PREPROCESSING & CLEANING ---
with st.expander("Langkah 1: Pembersihan Data Dasar", expanded=True):
    if st.button("🧼 Jalankan Cleaning"):
        with st.spinner("Membersihkan data..."):
            df_temp = st.session_state.df.copy()
            
            df_temp.columns = df_temp.columns.str.lower().str.replace(" ", "_")
            st.write("✅ Nama kolom diubah.")

            # Cleaning nama_amil
            if 'nama_amil' in df_temp.columns:
                df_temp['nama_amil'] = (
                    df_temp['nama_amil']
                    .astype(str)
                    .str.strip()
                    .str.replace(r"\s+", " ", regex=True)
                    .str.replace(u'\xa0', '', regex=False)
                    .str.title()
                )
                st.write("🧹 Kolom `nama_amil` dibersihkan dan dikapitalisasi.")

            missing_before = df_temp.isnull().sum().sum()
            df_temp.fillna(0, inplace=True)
            st.write(f"✅ {missing_before} nilai hilang diganti dengan 0.")

            dup_count = df_temp.duplicated().sum()
            if dup_count > 0:
                df_temp = df_temp.drop_duplicates()
                st.warning(f"⚠️ {dup_count} data duplikat dihapus.")
            else:
                st.info("ℹ️ Tidak ada data duplikat.")

            st.session_state.df_processed = df_temp
            st.success("Pembersihan data dasar selesai!")
            st.dataframe(st.session_state.df_processed.head())

    # 🧱 Tambahkan Hapus Kolom Langsung di Cleaning
    if 'drop_columns_cleaning' not in st.session_state:
        st.session_state.drop_columns_cleaning = []

    st.subheader("Hapus Kolom Tidak Relevan")
    all_cols = st.session_state.df_processed.columns.tolist()
    st.markdown("⚠️ **Catatan:** Jika Anda berencana melakukan klasterisasi berdasarkan _Nama Amil_, **jangan hapus kolom `nama_amil`** agar fitur tersebut bisa dijalankan.")

    selected_drop = st.multiselect(
        "Pilih kolom yang ingin dihapus:",
        all_cols,
        default=st.session_state.drop_columns_cleaning,
        key="drop_columns_multiselect"
    )

    if st.button("❌ Hapus Kolom Terpilih", key="drop_columns_button"):
        if 'nama_amil' in selected_drop:
            st.warning("⚠️ Anda memilih menghapus kolom `nama_amil`, kolom ini dibutuhkan jika ingin melakukan klasterisasi berdasarkan nama amil.")
        st.session_state.df_processed.drop(columns=selected_drop, inplace=True)
        st.success("Kolom berhasil dihapus.")
        st.dataframe(st.session_state.df_processed.head())


# --- OPTIONAL: EDA ---
with st.expander("Langkah 2: Exploratory Data Analysis (EDA)", expanded=True):
    df_eda = st.session_state.df_processed.copy()
    numeric_df = df_eda.select_dtypes(include=np.number)

    st.subheader("Statistik Deskriptif")
    st.dataframe(numeric_df.describe())

    col1, col2 = st.columns(2)

    # KIRI: Heatmap Korelasi
    with col1:
        st.markdown("Korelasi antar Variabel")
        corr = numeric_df.corr()
        import seaborn as sns
        import matplotlib.pyplot as plt

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax1)
        st.pyplot(fig1)

    # KANAN: Boxplot Outlier
    with col2:
        st.markdown("Visualisasi Outlier")
        selected_col = st.selectbox("Pilih kolom numerik:", numeric_df.columns)
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        sns.boxplot(y=numeric_df[selected_col], ax=ax2)
        ax2.set_title(f'Boxplot: {selected_col}')
        st.pyplot(fig2)


# --- STEP 2: OUTLIERS & FEATURE SELECTION ---
if st.session_state.df_processed is not None:
    with st.expander("Langkah 2: Penanganan Outlier (Opsional)"):
        df_current = st.session_state.df_processed.copy()

        # Outlier Handling
        st.subheader("Hapus Outlier")
        numeric_df = df_current.select_dtypes(include=np.number)
        if st.checkbox("Aktifkan penanganan outlier"):
            outlier_removed = numeric_df.copy()
            for col in numeric_df.columns:
                Q1 = numeric_df[col].quantile(0.25)
                Q3 = numeric_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers_count = ((numeric_df[col] < lower) | (numeric_df[col] > upper)).sum()
                if outliers_count > 0:
                    st.write(f"Kolom **{col}**: terdeteksi {outliers_count} outlier.")
                    outlier_removed = outlier_removed[(outlier_removed[col] >= lower) & (outlier_removed[col] <= upper)]
            
            if st.button("🗑️ Terapkan Penghapusan Outlier"):
                st.session_state.df_processed = df_current.loc[outlier_removed.index]
                st.success(f"Outlier berhasil dihapus!")
                st.dataframe(st.session_state.df_processed.head())
        
    # --- STEP 3: NORMALIZATION ---
    with st.expander("Langkah 3: Normalisasi Data"):
        st.write("Menyamakan skala data numerik menggunakan MinMaxScaler.")
        if st.button("⚖️ Lakukan Normalisasi"):
            numeric_df_final = st.session_state.df_processed.select_dtypes(include=np.number)
            if not numeric_df_final.empty:
                scaler = MinMaxScaler()
                st.session_state.scaled_data = scaler.fit_transform(numeric_df_final)
                st.session_state.feature_names = numeric_df_final.columns.tolist()
                st.success("Data berhasil dinormalisasi!")
                st.write("Data setelah Normalisasi (5 baris pertama):")
                st.dataframe(pd.DataFrame(st.session_state.scaled_data, columns=st.session_state.feature_names).head())
            else:
                st.error("Tidak ada data numerik untuk dinormalisasi.")
else:
    st.info("Selesaikan langkah 'Pembersihan Data Dasar' terlebih dahulu.")
