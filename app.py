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

# --- Antarmuka Input di Sidebar ---
with st.sidebar:
    # --- Blok CSS Kuat untuk Memadatkan Sidebar ---
    st.markdown("""
        <style>
            /* Target semua elemen di dalam sidebar */
            [data-testid="stSidebar"] * {
                line-height: 1.2; /* Kurangi jarak antar baris */
            }
            /* Target header di sidebar */
            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
                font-size: 18px; /* Perkecil ukuran header */
                margin-bottom: 0.5rem; /* Kurangi margin bawah */
            }
            /* Target label dari widget input angka */
            [data-testid="stSidebar"] .st-emotion-cache-1qg05j4 p {
                 font-size: 14px; /* Perkecil font label */
                 margin-bottom: 0.1rem; /* Kurangi jarak bawah label */
            }
            /* Target container dari setiap widget input */
            [data-testid="stSidebar"] .st-emotion-cache-ue6h4q {
                margin-bottom: 0.1rem; /* Kurangi jarak antar widget */
            }
            /* Target tombol utama di sidebar */
            [data-testid="stSidebar"] .stButton > button {
                margin-top: 1rem; /* Beri sedikit jarak atas untuk tombol */
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.header("Masukkan Indikator:")
    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(label=feature, step=0.01, format="%.4f")
    
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
    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1: st.success("Prediksi: **MERAIH OPINI WTP**")
        else: st.error("Prediksi: **TIDAK MERAIH OPINI WTP**")
    with col2:
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
