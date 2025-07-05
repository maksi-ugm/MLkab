# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

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
    Aplikasi ini menggunakan model **Random Forest** untuk memprediksi opini BPK.
    Anda dapat melakukan **prediksi tunggal** melalui sidebar di sebelah kiri, atau melakukan
    **uji massal (bulk test)** dengan mengunggah file CSV di bawah ini.
    """
)

# Memeriksa apakah artefak berhasil dimuat sebelum melanjutkan
if artifacts is None:
    st.error(
        "File 'artifacts_rf.pkl' tidak ditemukan. "
        "Pastikan file model sudah dibuat dan berada di repositori yang sama dengan aplikasi ini."
    )
    st.stop()

# Membuka artefak
model = artifacts['model']
imputer = artifacts['imputer']
scaler = artifacts['scaler']
features = artifacts['features']

# ==============================================================================
# Bagian 1: Prediksi Tunggal (di Sidebar)
# ==============================================================================
with st.sidebar:
    st.header("Prediksi Tunggal")
    st.write("Masukkan 11 Indikator Keuangan:")
    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(
            label=f"{feature}",
            step=0.01,
            format="%.4f",
            key=f"single_{feature}" # Kunci unik untuk input tunggal
        )
    
    if st.button("🚀 Lakukan Prediksi Tunggal"):
        input_df = pd.DataFrame([input_data])[features]
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        st.subheader("Hasil Prediksi:")
        if prediction[0] == 1:
            st.success("Diprediksi **MERAIH OPINI WTP**.")
        else:
            st.error("Diprediksi **TIDAK MERAIH OPINI WTP**.")
        
        st.write("Probabilitas:")
        st.write(f"WTP: {prediction_proba[0][1]:.2%}")
        st.write(f"Tidak WTP: {prediction_proba[0][0]:.2%}")


# ==============================================================================
# Bagian 2: Uji Massal / Bulk Test (di Halaman Utama)
# ==============================================================================
st.header("Uji Model dengan File CSV (Bulk Test)")
uploaded_file = st.file_uploader(
    "Unggah file CSV Anda di sini. Pastikan formatnya sama dengan data training (termasuk kolom 'WTP' untuk perbandingan).",
    type="csv"
)

if uploaded_file is not None:
    try:
        # Baca file yang diunggah
        df_bulk = pd.read_csv(uploaded_file, delimiter=';', decimal=',')
        st.success("File berhasil diunggah dan dibaca.")
        
        # Pisahkan fitur dan target asli
        X_bulk = df_bulk[features]
        y_true = df_bulk['WTP']

        # Lakukan pra-pemrosesan
        X_imputed = imputer.transform(X_bulk)
        X_scaled = scaler.transform(X_imputed)

        # Lakukan prediksi
        y_pred = model.predict(X_scaled)
        
        # Buat laporan hasil
        df_result = df_bulk.copy()
        df_result['Prediksi_Model'] = y_pred
        df_result['Hasil'] = (df_result['WTP'] == df_result['Prediksi_Model']).replace({True: '✅ Benar', False: '❌ Salah'})
        
        # Hitung akurasi
        accuracy = accuracy_score(y_true, y_pred)
        
        # Tampilkan ringkasan dan hasil detail
        st.subheader("Ringkasan Hasil Uji Massal")
        st.metric(label="Tingkat Akurasi pada File Uji", value=f"{accuracy:.2%}")
        
        st.subheader("Hasil Prediksi Detail")
        st.dataframe(df_result)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
        st.warning("Pastikan file CSV Anda memiliki delimiter ';' dan pemisah desimal ',', serta berisi semua kolom yang diperlukan.")
