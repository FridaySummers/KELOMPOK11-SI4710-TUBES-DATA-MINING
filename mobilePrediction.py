
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from joblib import load

# Load model dan data evaluasi
pipeline = load('mobile_price.joblib')
kmeans = load('kmeans_model.joblib')
report = load('classification_report.joblib')
conf_matrix = load('conf_matrix.joblib')

selected_features = [
    'battery_power', 'blue', 'dual_sim', 'fc', 'four_g',
    'int_memory', 'm_dep', 'mobile_wt', 'pc', 'wifi'
]

# UI Awal
st.set_page_config(page_title="Prediksi & Segmentasi Ponsel", layout="centered")
st.title("📱 Aplikasi Prediksi Harga & Segmentasi Ponsel")

# Navigasi tab
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Prediksi Harga", "📊 Prediksi Klaster", "📈 Visualisasi Performa", "🔹 Visualisasi Performa Clustering"])

# ============================ TAB 1 ============================
with tab1:
    st.subheader("🔍 Prediksi Kisaran Harga Ponsel (Supervised Learning)")

    with st.expander("ℹ️ Petunjuk Pengisian"):
        st.markdown("""
        - **Kapasitas Baterai:** Daya baterai dalam mAh.
        - **RAM:** Ukuran RAM dalam MB.
        - **Dual SIM, 4G, Wi-Fi:** Pilih `Ya` atau `Tidak`.
        """)

    col1, col2 = st.columns(2)
    with col1:
        battery_power = st.slider("Kapasitas Baterai (mAh)", 500, 2000, 1000)
        ram = st.slider("RAM (MB)", 256, 4000, 1500)
        fc = st.slider("Kamera Depan (MP)", 0, 20, 5)
        pc = st.slider("Kamera Belakang (MP)", 0, 20, 10)
        int_memory = st.slider("Memori Internal (GB)", 2, 128, 32)

    with col2:
        mobile_wt = st.slider("Berat Ponsel (gram)", 80, 250, 150)
        m_dep = st.slider("Ketebalan Ponsel (cm)", 0.1, 1.0, 0.5)
        dual_sim = st.selectbox("Dual SIM", ["Tidak", "Ya"])
        four_g = st.selectbox("4G", ["Tidak", "Ya"])
        wifi = st.selectbox("Wi-Fi", ["Tidak", "Ya"])

    binary_map = {"Tidak": 0, "Ya": 1}
    dual_sim = binary_map[dual_sim]
    four_g = binary_map[four_g]
    wifi = binary_map[wifi]

    input_data = np.array([[battery_power, ram, fc, pc, int_memory,
                            mobile_wt, m_dep, dual_sim, four_g, wifi]])

    if st.button("🔎 Prediksi Harga"):
        scaler = load('scaler.joblib')  # Pastikan scaler.joblib sudah ada
        input_scaled = scaler.transform(input_data)
        price_class = pipeline.predict(input_scaled)[0]
        kategori = ["💸 Murah", "💰 Menengah", "💎 Mahal", "👑 Premium"]
        warna = ["green", "orange", "blue", "red"]

        st.markdown(f"<h3 style='color:{warna[price_class]};'>Kategori Harga: {kategori[price_class]}</h3>", unsafe_allow_html=True)
        st.write("📥 Data yang Anda masukkan:")
        st.dataframe(pd.DataFrame(input_data, columns=selected_features))

# ============================ TAB 2 ============================
with tab2:
    st.subheader("📊 Prediksi Klaster Ponsel (Unsupervised Learning)")

    input_data = np.array([[battery_power, ram, fc, pc, int_memory,
                            mobile_wt, m_dep, dual_sim, four_g, wifi]])

    cluster = kmeans.predict(input_data)[0]

    cluster_desc = {
        0: "📱 Ponsel ringan dengan spesifikasi dasar",
        1: "⚡ Ponsel mid-range dengan kamera baik",
        2: "📸 Ponsel dengan fitur multimedia kuat",
        3: "🚀 Ponsel flagship dengan performa tinggi"
    }

    st.success(f"🧠 Klaster: {cluster}")
    st.info(f"📄 Deskripsi Klaster: {cluster_desc.get(cluster, 'Tidak dikenali')}")

    st.write("📥 Data yang Anda masukkan:")
    st.dataframe(pd.DataFrame(input_data, columns=selected_features))

# ============================ TAB 3 ============================
with tab3:
    st.subheader("📈 Evaluasi Performa Model Klasifikasi")

    st.markdown("### 🔹 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(pd.DataFrame(conf_matrix), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.markdown("### 🔹 Classification Report")
    st.json(report)

# ============================ TAB 4 ============================
with tab4:
    st.subheader("🔹 Visualisasi Performa Clustering (KMeans & Elbow)")

    st.markdown("#### Elbow Method")
    inertia = []
    range_k = range(1, 11)
    scaler = load('scaler.joblib')
    X_cluster = None
    try:
        df = pd.read_csv("train.csv")
        selected_features = [
            'battery_power', 'blue', 'dual_sim', 'fc', 'four_g',
            'int_memory', 'm_dep', 'mobile_wt', 'pc', 'wifi'
        ]
        X_cluster = scaler.transform(df[selected_features])
        for k in range_k:
            kmeans_tmp = KMeans(n_clusters=k, random_state=42)
            kmeans_tmp.fit(X_cluster)
            inertia.append(kmeans_tmp.inertia_)
        fig1, ax1 = plt.subplots()
        ax1.plot(range_k, inertia, marker='o')
        ax1.set_xlabel('Jumlah Cluster (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method for Optimal k')
        st.pyplot(fig1)
    except Exception as e:
        st.warning(f"Gagal menampilkan Elbow Method: {e}")

    st.markdown("#### Visualisasi Cluster (PCA 2D)")
    try:
        if X_cluster is not None:
            kmeans = load('kmeans_model.joblib')
            pca = load('pca_model.joblib')
            clusters = kmeans.predict(X_cluster)
            X_pca = pca.transform(X_cluster)
            pca_df = pd.DataFrame(X_pca, columns=['PCA 1', 'PCA 2'])
            pca_df['Cluster'] = clusters

            fig2, ax2 = plt.subplots(figsize=(8,6))
            sns.scatterplot(x='PCA 1', y='PCA 2', hue='Cluster', data=pca_df, palette='viridis', s=80, ax=ax2)
            ax2.set_title('KMeans Clustering Visualization (PCA)')
            st.pyplot(fig2)
        else:
            st.warning("Data cluster tidak tersedia untuk visualisasi PCA.")
    except Exception as e:
        st.warning(f"Gagal menampilkan visualisasi cluster: {e}")

# Footer
st.markdown("---")
st.caption("🧪 Aplikasi ini dikembangkan untuk mendukung penelitian segmentasi ponsel berdasarkan spesifikasi teknis dan kisaran harga menggunakan pendekatan supervised dan unsupervised learning.")
