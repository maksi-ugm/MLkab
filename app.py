# app.py
import streamlit as st
import pandas as pd
import joblib

# --- Fungsi untuk memuat artefak ---
@st.cache_resource
def load_artifacts():
    """Memuat model, imputer, dan scaler yang sudah dilatih."""
    artifacts = joblib.load('artifacts_rf.pkl')
    return artifacts

# --- Memuat artefak di awal ---
try:
    artifacts = load_artifacts()
    model = artifacts['model']
    imputer = artifacts['imputer']
    scaler = artifacts['scaler']
    features = artifacts['features']
    print("Artefak berhasil dimuat.")
except FileNotFoundError:
    st.error("File 'artifacts_rf.pkl' tidak ditemukan. Harap pastikan file tersebut ada di repository GitHub Anda.")
    st.stop()


# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Prediksi WTP", page_icon="📈", layout="wide")
st.title("📈 Aplikasi Prediksi Willingness to Pay (WTP)")
st.write("""
Aplikasi ini menggunakan model **Random Forest** untuk memprediksi apakah seorang responden memiliki *Willingness to Pay* (WTP) berdasarkan 11 indikator keuangan.
""")

# --- Antarmuka Input ---
with st.sidebar:
    st.header("Masukkan Nilai Indikator:")
    input_data = {}
    for feature in features:
        # Menggunakan number_input untuk semua fitur
        input_data[feature] = st.number_input(
            label=f"{feature}",
            step=0.01,
            format="%.4f"
        )

# --- Tombol Prediksi ---
if st.sidebar.button("🚀 Lakukan Prediksi"):
    # 1. Ubah input menjadi DataFrame
    input_df = pd.DataFrame([input_data])
    
    st.subheader("Data Input Anda:")
    st.dataframe(input_df)

    # 2. Lakukan pra-pemrosesan pada data input
    # Urutkan kolom agar sesuai dengan urutan saat training
    input_df = input_df[features]
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)

    # 3. Lakukan prediksi
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # 4. Tampilkan hasil
    st.subheader("Hasil Prediksi:")
    if prediction[0] == 1:
        st.success("Responden diprediksi **WTP** (Willingness to Pay).")
    else:
        st.error("Responden diprediksi **Non-WTP**.")

    # Tampilkan probabilitas
    st.write("Tingkat Keyakinan Model:")
    proba_df = pd.DataFrame({
        'Kelas': ['Non-WTP (0)', 'WTP (1)'],
        'Probabilitas': prediction_proba[0]
    })
    st.dataframe(proba_df.style.format({'Probabilitas': "{:.2%}"}))

else:
    st.info("Silakan masukkan nilai pada sidebar di sebelah kiri dan klik tombol prediksi.")
