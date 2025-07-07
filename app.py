# app.py
import streamlit as st
import pandas as pd
import joblib
import altair as alt

# --- Konfigurasi Halaman & Fungsi Pemuatan ---
st.set_page_config(page_title="Dashboard Diagnostik Opini WTP", page_icon="🩺", layout="wide")

@st.cache_resource
def load_artifacts():
    try:
        return joblib.load('artifacts_hybrid.pkl')
    except FileNotFoundError:
        return None

artifacts = load_artifacts()

if artifacts is None:
    st.error("File 'artifacts_hybrid.pkl' tidak ditemukan. Harap jalankan skrip training hybrid terlebih dahulu.")
    st.stop()

# Membuka semua artefak
model_rf = artifacts['model_rf']
model_lr = artifacts['model_lr']
imputer = artifacts['imputer']
scaler = artifacts['scaler']
features = artifacts['features']
benchmark = artifacts['benchmark']

# --- Judul dan Panduan Pengguna ---
st.title("🩺 Dashboard Diagnostik Prediksi Opini WTP")

with st.expander("Lihat Cara Membaca Hasil Analisis Ini"):
    st.write("""
        - **Prediksi & Keyakinan**: Menunjukkan prediksi opini (WTP/Tidak WTP) dari model dan seberapa yakin model dengan prediksi tersebut.
        - **Analisis Faktor Pendorong**: Bagian terpenting. Ini menunjukkan **indikator mana yang paling berpengaruh** terhadap hasil prediksi.
            - **Tingkat Pengaruh**: Seberapa besar dampak sebuah indikator. Semakin panjang barnya, semakin besar pengaruhnya.
            - **Arah Pengaruh**:
                - 🟩 **Positif (Mendorong ke Arah WTP)**: Nilai indikator ini sudah baik dan membantu meningkatkan peluang WTP.
                - 🟥 **Negatif (Menjauhkan dari WTP)**: Nilai indikator ini menjadi **penghambat utama**. Ini adalah area yang perlu menjadi fokus perbaikan.
        - **Rekomendasi Kebijakan**: Saran praktis yang dihasilkan otomatis berdasarkan faktor pendorong negatif yang paling signifikan.
    """)

# --- Antarmuka Input di Sidebar dengan Layout Kolom ---
with st.sidebar:
    st.header("Masukkan Indikator")
    
    # Buat dua kolom di dalam sidebar
    col1, col2 = st.columns(2)
    
    input_data = {}
    
    # Bagi 11 fitur ke dalam dua kolom (6 di kiri, 5 di kanan)
    for i, feature in enumerate(features):
        if i < 6:
            with col1:
                input_data[feature] = st.number_input(label=feature, step=0.01, format="%.4f")
        else:
            with col2:
                input_data[feature] = st.number_input(label=feature, step=0.01, format="%.4f")
    
    st.markdown("---") # Garis pemisah
    predict_button = st.button("🚀 Lakukan Analisis", type="primary", use_container_width=True)

# --- Logika Utama Aplikasi (Tidak ada perubahan di sini) ---
if predict_button:
    # Pra-pemrosesan & Prediksi
    input_df = pd.DataFrame([input_data])[features]
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)
    prediction = model_rf.predict(input_scaled)[0]
    prediction_proba = model_rf.predict_proba(input_scaled)[0]

    st.header("Hasil Prediksi & Tingkat Keyakinan")
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        if prediction == 1: st.success("Prediksi: **MERAIH OPINI WTP**")
        else: st.error("Prediksi: **TIDAK MERAIH OPINI WTP**")
    with res_col2:
        st.metric(label="Keyakinan Model (Probabilitas WTP)", value=f"{prediction_proba[1]:.2%}")

    st.divider()
    
    st.header("Analisis Faktor Pendorong Utama (Key Drivers)")
    diagnostics_df = pd.DataFrame({
        'Indikator': features,
        'Tingkat Pengaruh': model_rf.feature_importances_,
        'Arah Pengaruh (Koefisien LR)': model_lr.coef_[0],
    }).sort_values('Tingkat Pengaruh', ascending=False).head(7)
    diagnostics_df['Arah Pengaruh Label'] = diagnostics_df['Arah Pengaruh (Koefisien LR)'].apply(lambda x: 'Positif' if x > 0 else 'Negatif')

    chart = alt.Chart(diagnostics_df).mark_bar().encode(
        x=alt.X('Tingkat Pengaruh:Q', title='Tingkat Pengaruh'),
        y=alt.Y('Indikator:N', sort='-x', title='Indikator'),
        color=alt.Color('Arah Pengaruh Label:N', scale=alt.Scale(domain=['Positif', 'Negatif'], range=['#2ECC71', '#E74C3C']), title="Arah"),
        tooltip=['Indikator', alt.Tooltip('Tingkat Pengaruh:Q', format='.3f')]
    ).properties(title='Indikator Paling Berpengaruh')
    st.altair_chart(chart, use_container_width=True)

    st.header("Rekomendasi Kebijakan Otomatis")
    negative_drivers = diagnostics_df[diagnostics_df['Arah Pengaruh Label'] == 'Negatif']
    recommendations = []
    if not negative_drivers.empty:
        st.write("Area prioritas yang perlu mendapat perhatian:")
        for _, row in negative_drivers.iterrows():
            indikator = row['Indikator']
            if 'Kemandirian Keuangan' in indikator: recommendations.append(f"**{indikator}:** Tingkatkan PAD melalui program intensifikasi/ekstensifikasi.")
            elif 'Solvabilitas Anggaran' in indikator: recommendations.append(f"**{indikator}:** Lakukan efisiensi belanja dan tinjau ulang proyeksi pendapatan.")
            elif 'Solvabilitas Jangka Panjang' in indikator: recommendations.append(f"**{indikator}:** Tinjau struktur aset dan kewajiban jangka panjang.")
            elif 'Efektifitas' in indikator: recommendations.append(f"**{indikator}:** Optimalisasi potensi pajak dan retribusi daerah.")
    if recommendations:
        for rec in recommendations: st.warning(f"👉 {rec}")
    else:
        st.success("✅ Selamat! Semua faktor pendorong utama menunjukkan pengaruh positif.")

    st.divider()
    st.page_link("https://maksikeuda.streamlit.app/", label="Buka Dashboard Perbandingan Pemda", icon="📊")
else:
    st.info("Silakan masukkan nilai pada sidebar dan klik tombol 'Lakukan Analisis' untuk memulai.")
