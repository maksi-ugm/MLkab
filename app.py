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
        # Ganti nama file menjadi artifacts_hybrid.pkl
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

# PRIORITAS #3: Panduan Pengguna
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

# --- Antarmuka Input di Sidebar ---
with st.sidebar:
    st.header("Masukkan Indikator Keuangan:")
    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(label=feature, step=0.01, format="%.4f")
    
    predict_button = st.button("🚀 Lakukan Analisis Diagnostik", type="primary")

# --- Logika Utama Aplikasi ---
if predict_button:
    # 1. Pra-pemrosesan & Prediksi (menggunakan model RF)
    input_df = pd.DataFrame([input_data])[features]
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)
    prediction = model_rf.predict(input_scaled)[0]
    prediction_proba = model_rf.predict_proba(input_scaled)[0]

    st.header("Hasil Prediksi & Tingkat Keyakinan")
    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.success("Prediksi: **MERAIH OPINI WTP**")
        else:
            st.error("Prediksi: **TIDAK MERAIH OPINI WTP**")
    with col2:
        st.metric(label="Keyakinan Model (Probabilitas WTP)", value=f"{prediction_proba[1]:.2%}")

    st.divider()

    # --- ANALISIS FAKTOR PENDORONG (HYBRID APPROACH) ---
    st.header("Analisis Faktor Pendorong Utama (Key Drivers)")

    # Gabungkan informasi dari kedua model
    diagnostics_df = pd.DataFrame({
        'Indikator': features,
        'Tingkat Pengaruh': model_rf.feature_importances_,
        'Arah Pengaruh (Koefisien LR)': model_lr.coef_[0],
        'Nilai Input': input_df.iloc[0],
        'Nilai Benchmark': benchmark
    })
    
    # Tambahkan label kualitatif
    diagnostics_df['Arah Pengaruh Label'] = diagnostics_df['Arah Pengaruh (Koefisien LR)'].apply(lambda x: 'Positif' if x > 0 else 'Negatif')
    diagnostics_df = diagnostics_df.sort_values('Tingkat Pengaruh', ascending=False).head(7)

    # Visualisasi dengan Altair
    chart = alt.Chart(diagnostics_df).mark_bar().encode(
        x=alt.X('Tingkat Pengaruh:Q', title='Tingkat Pengaruh (Importance)'),
        y=alt.Y('Indikator:N', sort='-x', title='Indikator Keuangan'),
        color=alt.Color('Arah Pengaruh Label:N',
                        scale=alt.Scale(domain=['Positif', 'Negatif'], range=['#2ECC71', '#E74C3C']),
                        title="Arah Pengaruh"),
        tooltip=[
            alt.Tooltip('Indikator:N'),
            alt.Tooltip('Nilai Input:Q', format='.4f', title='Nilai Anda'),
            alt.Tooltip('Nilai Benchmark:Q', format='.4f', title='Nilai Benchmark'),
            alt.Tooltip('Tingkat Pengaruh:Q', format='.3f'),
        ]
    ).properties(
        title='Indikator Paling Berpengaruh Terhadap Prediksi'
    )
    st.altair_chart(chart, use_container_width=True)

    # --- REKOMENDASI OTOMATIS BERDASARKAN FAKTOR NEGATIF ---
    st.header("Rekomendasi Kebijakan Otomatis")

    # Filter hanya untuk pendorong negatif
    negative_drivers = diagnostics_df[diagnostics_df['Arah Pengaruh Label'] == 'Negatif']
    recommendations = []

    if not negative_drivers.empty:
        st.write("Berdasarkan analisis, berikut adalah area prioritas yang perlu mendapat perhatian:")
        for index, row in negative_drivers.iterrows():
            indikator = row['Indikator']
            if 'Kemandirian Keuangan' in indikator:
                recommendations.append(f"**{indikator}:** Tingkatkan Pendapatan Asli Daerah (PAD) melalui program intensifikasi dan ekstensifikasi.")
            elif 'Solvabilitas Anggaran' in indikator:
                 recommendations.append(f"**{indikator}:** Lakukan efisiensi belanja, terutama belanja operasional, dan tinjau kembali proyeksi pendapatan agar lebih realistis.")
            elif 'Solvabilitas Jangka Panjang' in indikator:
                 recommendations.append(f"**{indikator}:** Perlu peninjauan atas struktur aset dan kewajiban jangka panjang.")
            elif 'Efektifitas' in indikator:
                 recommendations.append(f"**{indikator}:** Tinjau kembali potensi pajak dan retribusi daerah yang belum optimal.")
    
    if recommendations:
        for rec in recommendations:
            st.warning(f"👉 {rec}")
    else:
        st.success("✅ Selamat! Semua faktor pendorong utama menunjukkan pengaruh positif. Pertahankan kinerja yang sudah baik.")

    # --- Link ke Dashboard Eksternal ---
    st.divider()
    st.page_link("https://maksikeuda.streamlit.app/", label="Buka Dashboard Perbandingan Pemda", icon="📊")

else:
    st.info("Silakan masukkan nilai pada sidebar dan klik tombol 'Lakukan Analisis Diagnostik' untuk memulai.")
