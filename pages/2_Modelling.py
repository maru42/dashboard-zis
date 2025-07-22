import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Modelling", layout="wide")
st.title("Clustering")
st.markdown("Tentukan jumlah cluster (K) yang optimal dan jalankan algoritma K-Means.")

# Validasi awal
if 'df_processed' not in st.session_state:
    st.warning("⚠️ Data belum tersedia. Silakan lakukan preprocessing terlebih dahulu.")
    st.stop()

df_raw = st.session_state.df_processed.copy()
selected_cols = ['zakat_fitrah_beras_(kg)', 'zakat_fitrah_uang',
                 'zakat_mall', 'fidyah', 'infaq_/_shodaqoh']

st.subheader("📊 Pilih Mode Klasterisasi")
st.markdown("""
**Pilih mode klasterisasi terlebih dahulu.**
- **Per Baris Data (Transaksi):** Setiap transaksi dianggap unik.
- **Per Nama Amil:** Data digabung berdasarkan nama amil, total ZIS dijumlahkan, lalu dikelompokkan.
""")

cluster_mode = st.radio(
    "Ingin klasterisasi berdasarkan apa?",
    ('Per Baris Data (Transaksi)', 'Per Nama Amil'),
    help="Per Baris Data akan memperlakukan setiap baris sebagai entitas unik, sedangkan Per Nama Amil akan menjumlahkan total ZIS berdasarkan nama amil terlebih dahulu."
)

# Siapkan data berdasarkan mode
if cluster_mode == 'Per Nama Amil':
    if 'nama_amil' not in df_raw.columns:
        st.error("Kolom 'nama_amil' tidak tersedia. Pastikan tidak dihapus pada tahap preprocessing.")
        st.stop()
    df_grouped = df_raw.groupby('nama_amil')[selected_cols].sum().reset_index()
    scaled_data = MinMaxScaler().fit_transform(df_grouped[selected_cols])
    df_for_modelling = df_grouped
else:
    scaled_data = MinMaxScaler().fit_transform(df_raw[selected_cols])
    df_for_modelling = df_raw

# Simpan ke session
st.session_state.scaled_data = scaled_data
st.session_state.feature_names = selected_cols
st.session_state.df_for_modelling = df_for_modelling

max_k = min(10, len(scaled_data))

# Debug Sidebar
st.sidebar.subheader("🔍 Debug Info")
st.sidebar.write(f"Jumlah baris data: {scaled_data.shape[0]}")
st.sidebar.write(f"Jumlah fitur: {scaled_data.shape[1]}")
st.sidebar.subheader("Statistik Data Scaled")
st.sidebar.write(pd.DataFrame(scaled_data).describe())

# --- STEP 4: Tentukan Jumlah K ---
with st.expander("Langkah 4: Tentukan Jumlah Cluster (K) Optimal", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Elbow Method")
        if st.button("📈 Jalankan Elbow Method"):
            with st.spinner("Menghitung inertia..."):
                distortions = []
                max_k = min(10, len(scaled_data))
                K_range = range(1, max_k + 1)

                if np.isnan(scaled_data).any() or np.isinf(scaled_data).any():
                    st.error("⚠️ Data mengandung nilai NaN atau Infinity! Periksa kembali preprocessing.")
                    st.stop()

                for k in K_range:
                    km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
                    km.fit(scaled_data)
                    distortions.append(km.inertia_)
                    st.write(f"K={k}: Inertia = {km.inertia_:.2f}")

                df_elbow = pd.DataFrame({'Jumlah Cluster (K)': K_range, 'Inertia': distortions})
                st.dataframe(df_elbow.style.format({'Inertia': '{:.2f}'}))

                fig = px.line(df_elbow, x='Jumlah Cluster (K)', y='Inertia', markers=True,
                              title="Elbow Method untuk Menentukan K")
                fig.update_traces(line=dict(color='royalblue', width=3))
                fig.update_layout(xaxis_title="Jumlah Cluster K", yaxis_title="Inertia (WCSS)", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Silhouette Score")
        if st.button("📏 Hitung Silhouette Score"):
            with st.spinner("Menghitung score..."):
                scores = {}
                for k in range(2, max_k + 1):
                    try:
                        model = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
                        labels = model.fit_predict(scaled_data)
                        score = silhouette_score(scaled_data, labels)
                        scores[k] = score
                    except ValueError:
                        scores[k] = -1
                        st.error(f"Tidak bisa hitung silhouette untuk K={k}")

                df_score = pd.DataFrame({
                    'Jumlah Cluster (K)': list(scores.keys()),
                    'Silhouette Score': list(scores.values())
                })

                st.dataframe(df_score.style.format({'Silhouette Score': '{:.4f}'}))

                valid_scores = {k: v for k, v in scores.items() if v >= 0}
                if valid_scores:
                    best_k = max(valid_scores, key=valid_scores.get)
                    st.success(f"✅ K terbaik menurut Silhouette Score: **{best_k}** (Score: {scores[best_k]:.4f})")
                    fig = px.bar(df_score, x='Jumlah Cluster (K)', y='Silhouette Score',
                                 title="Silhouette Score untuk Setiap K")
                    fig.update_traces(marker_color='#4CAF50')
                    st.plotly_chart(fig, use_container_width=True)

# --- STEP 5: Jalankan Clustering ---
with st.expander("Langkah 5: Jalankan Clustering", expanded=True):
    st.markdown("### Pilih Nilai K")
    k_value = st.slider("Pilih nilai K final untuk clustering", 2, max_k, 3, key="k_slider_final")

    if st.button("🚀 Jalankan K-Means Clustering", key="run_clustering"):
        with st.spinner("Membuat cluster..."):
            model = KMeans(n_clusters=k_value, init='k-means++', random_state=42, n_init=10)
            cluster_labels = model.fit_predict(scaled_data)

            result_df = df_for_modelling.copy()
            if len(result_df) != len(cluster_labels):
                result_df = result_df.reset_index(drop=True)

            result_df['cluster'] = cluster_labels
            st.session_state.result_df = result_df
            st.session_state.model = model

            st.success(f"Clustering dengan K={k_value} berhasil! Lihat hasilnya di halaman 'Hasil Analisis'.")

            cluster_dist = result_df['cluster'].value_counts().sort_index()
            cluster_dist_df = pd.DataFrame({
                'Cluster': cluster_dist.index,
                'Jumlah Sampel': cluster_dist.values
            })

            st.subheader("Distribusi Cluster")
            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(cluster_dist_df)

            with col2:
                fig = px.pie(cluster_dist_df, values='Jumlah Sampel', names='Cluster',
                             title=f"Distribusi Cluster (K={k_value})", hole=0.3)
                st.plotly_chart(fig, use_container_width=True)

# Preview akhir
if 'result_df' in st.session_state:
    st.subheader("Preview Data Hasil Clustering")
    st.dataframe(st.session_state.result_df.head())
