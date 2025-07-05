# app.py
import streamlit as st
import pandas as pd
import joblib

# --- Fungsi untuk memuat artefak (model, scaler, dll.) ---
@st.cache_resource
def load_artifacts():
    """Memuat model, imputer, dan scaler yang sudah dilatih."""
    try:
        artifacts = joblib.load('artifacts_rf.pkl')
        return artifacts
    except FileNotFoundError:
        return None

# --- Memuat artefak di awal ---
artifacts = load_artifacts()

if artifacts is None:
    st.error(
        "File 'artifacts_rf.pkl' tidak ditemukan. "
        "Pastikan file model sudah dibuat dan berada di repositori yang sama dengan aplikasi ini."
    )
    st.stop()

model = artifacts['model']
imputer = artifacts['imputer']
scaler = artifacts['scaler']
features = artifacts['features']

# --- Konfigurasi Halaman Aplikasi ---
st.set_page_config(
    page_title="Prediksi Opini WTP Pemda",
    page_icon="🏆",
    layout="wide"
)

# --- Judul dan Deskripsi Aplikasi ---
st.title("🏆 Aplikasi Prediksi Opini Wajar Tanpa Pengecualian (WTP)")
st.write(
    """
    Selamat datang di aplikasi prediksi opini BPK. Aplikasi ini menggunakan model *machine learning*
    (**Random Forest**) untuk memprediksi apakah suatu pemerintah daerah (kabupaten)
    berpotensi meraih opini **Wajar Tanpa Pengecualian (WTP)** berdasarkan 11 indikator keuangannya.
    """
)

# --- Antarmuka Input di Sidebar ---
with st.sidebar:
    st.header("Masukkan Indikator Keuangan:")
    input_data = {}
    # Membuat input field untuk setiap fitur/indikator
    for feature in features:
        input_data[feature] = st.number_input(
            label=f"{feature}",
            step=0.01,
            format="%.4f"
        )

# --- Tombol untuk Melakukan Prediksi ---
if st.sidebar.button("🚀 Lakukan Prediksi"):
    # 1. Ubah data input menjadi DataFrame
    input_df = pd.DataFrame([input_data])
    
    st.subheader("Data Indikator yang Dimasukkan:")
    st.dataframe(input_df)

    # 2. Lakukan pra-pemrosesan pada data input (imputasi & scaling)
    # Pastikan urutan kolom sesuai dengan saat training
    input_df = input_df[features]
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)

    # 3. Lakukan prediksi menggunakan model
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # 4. Tampilkan hasil prediksi
    st.subheader("Hasil Prediksi Opini:")
    if prediction[0] == 1:
        st.success("Pemerintah Daerah diprediksi **MERAIH OPINI WTP** (Wajar Tanpa Pengecualian).")
    else:
        st.error("Pemerintah Daerah diprediksi **TIDAK MERAIH OPINI WTP**.")

    # Tampilkan probabilitas sebagai tingkat keyakinan
    st.write("Tingkat Keyakinan Prediksi:")
    proba_df = pd.DataFrame({
        'Opini': ['Tidak WTP (0)', 'WTP (1)'],
        'Probabilitas': prediction_proba[0]
    })
    st.dataframe(
        proba_df.style.format({'Probabilitas': "{:.2%}"}),
        use_container_width=True
    )
else:
    # Pesan default saat halaman pertama kali dibuka
    st.info("Silakan masukkan nilai indikator keuangan pada sidebar di sebelah kiri dan klik tombol prediksi untuk melihat hasilnya.")
