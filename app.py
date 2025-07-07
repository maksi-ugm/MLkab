# app.py
import streamlit as st
import pandas as pd
import joblib
import altair as alt

# --- Konfigurasi Halaman & Fungsi Pemuatan ---
st.set_page_config(page_title="Analisis & Prediksi Opini WTP", page_icon="💡", layout="wide")

@st.cache_resource
def load_artifacts():
    try:
        return joblib.load('artifacts_rf.pkl')
    except FileNotFoundError:
        return None

artifacts = load_artifacts()

if artifacts is None:
    st.error("File 'artifacts_rf.pkl' tidak ditemukan. Harap jalankan skrip training terlebih dahulu.")
    st.stop()

# Membuka semua artefak
model = artifacts['model']
imputer = artifacts['imputer']
scaler = artifacts['scaler']
features = artifacts['features']
benchmark = artifacts['benchmark']

# --- Judul dan Deskripsi ---
st.title("💡 Dashboard Analisis & Prediksi Opini WTP")
st.write("""
Aplikasi ini tidak hanya memprediksi **Opini Wajar Tanpa Pengecualian (WTP)**, tetapi juga menganalisis **faktor pendorong utama**, membandingkannya dengan **tolok ukur (benchmark)**, dan memberikan **rekomendasi kebijakan** secara otomatis.
""")

# --- Antarmuka Input di Sidebar ---
with st.sidebar:
    st.header("Masukkan Indikator Keuangan:")
    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(label=feature, step=0.01, format="%.4f")
    
    predict_button = st.button("🚀 Analisis & Prediksi", type="primary")

# --- Logika Utama Aplikasi ---
if predict_button:
    # 1. Pra-pemrosesan & Prediksi
    input_df = pd.DataFrame([input_data])[features]
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]

    # --- Tampilan Hasil Prediksi ---
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

    # --- PRIORITAS #1: FAKTOR PENDORONG (KEY DRIVERS) ---
    st.header("Analisis Faktor Pendorong Utama")
    
    # Membuat DataFrame untuk feature importance
    importance_df = pd.DataFrame({
        'Indikator': features,
        'Nilai Penting': model.feature_importances_
    }).sort_values('Nilai Penting', ascending=False).head(7) # Ambil 7 teratas
    
    # Visualisasi dengan Altair untuk kontrol lebih
    chart = alt.Chart(importance_df).mark_bar().encode(
        x=alt.X('Nilai Penting:Q', title='Tingkat Pengaruh'),
        y=alt.Y('Indikator:N', sort='-x', title='Indikator Keuangan'),
        tooltip=['Indikator', 'Nilai Penting']
    ).properties(
        title='Indikator Paling Berpengaruh Terhadap Prediksi'
    )
    st.altair_chart(chart, use_container_width=True)

    # --- PRIORITAS #2: KONTEKS DENGAN TOLOK UKUR (BENCHMARKING) ---
    st.header("Perbandingan dengan Tolok Ukur (Benchmark)")
    
    benchmark_df = pd.DataFrame({
        'Indikator': features,
        'Nilai Input': input_df.iloc[0],
        'Rata-rata Benchmark': benchmark
    }).reset_index(drop=True)
    
    st.dataframe(
        benchmark_df.style.format({
            'Nilai Input': '{:.4f}',
            'Rata-rata Benchmark': '{:.4f}'
        }).highlight_max(subset=['Nilai Input', 'Rata-rata Benchmark'], color='lightgreen', axis=1)
          .highlight_min(subset=['Nilai Input', 'Rata-rata Benchmark'], color='#ffcccb', axis=1),
        use_container_width=True
    )

    # --- PRIORITAS #3: REKOMENDASI OTOMATIS BERBASIS ATURAN ---
    st.header("Rekomendasi Kebijakan Otomatis")

    # Ambil 3 pendorong teratas
    top_drivers = importance_df['Indikator'].head(3).tolist()
    recommendations = []

    # Aturan sederhana (bisa dikembangkan lebih lanjut)
    if 'Kemandirian Keuangan' in top_drivers and input_df['Kemandirian Keuangan'].iloc[0] < benchmark['Kemandirian Keuangan']:
        recommendations.append("**Kemandirian Keuangan Rendah:** Prioritaskan program intensifikasi dan ekstensifikasi Pendapatan Asli Daerah (PAD) untuk meningkatkan kemandirian fiskal.")
    
    if 'Solvabilitas Anggaran' in top_drivers and input_df['Solvabilitas Anggaran'].iloc[0] < benchmark['Solvabilitas Anggaran']:
        recommendations.append("**Solvabilitas Anggaran Kritis:** Lakukan efisiensi belanja, terutama belanja operasional, dan tinjau kembali proyeksi pendapatan agar lebih realistis.")

    if 'Solvabilitas Jangka Panjang' in top_drivers and input_df['Solvabilitas Jangka Panjang'].iloc[0] < benchmark['Solvabilitas Jangka Panjang']:
        recommendations.append("**Solvabilitas Jangka Panjang Terancam:** Perlu peninjauan atas struktur aset dan kewajiban jangka panjang. Pertimbangkan restrukturisasi utang jika memungkinkan.")

    if 'Rasio Efektifitas Pengelolaan Pendapatan' in top_drivers and input_df['Rasio Efektifitas Pengelolaan Pendapatan'].iloc[0] < benchmark['Rasio Efektifitas Pengelolaan Pendapatan']:
         recommendations.append("**Efektivitas PAD Perlu Ditingkatkan:** Tinjau kembali potensi pajak dan retribusi daerah yang belum optimal. Manfaatkan teknologi untuk mempermudah pembayaran dan pengawasan.")

    if recommendations:
        for rec in recommendations:
            st.warning(rec)
    else:
        st.success("Secara umum, indikator-indikator kunci sudah menunjukkan performa yang baik. Pertahankan!")

    # --- Link ke Dashboard Eksternal ---
    st.divider()
    st.info("Untuk analisis perbandingan yang lebih mendalam antar pemerintah daerah, kunjungi dashboard eksternal kami.")
    st.page_link("https://maksikeuda.streamlit.app/", label="Buka Dashboard Perbandingan Pemda", icon="📊")

else:
    st.info("Silakan masukkan nilai pada sidebar dan klik tombol 'Analisis & Prediksi' untuk memulai.")
