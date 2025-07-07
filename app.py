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

# --- DICTIONARY UNTUK TOOLTIPS ---
tooltip_texts = {
    "Solvabilitas Jangka Pendek": "Aset lancar dibagi dengan kewajiban jangka pendek pada tahun yang bersangkutan.",
    "Solvabilitas Jangka Panjang": "Total Aset Tetap (nilai buku) dibagi dengan kewajiban jangka panjang pada tahun yang bersangkutan.",
    "Kemandirian Keuangan": "Total Pendapatan Asli Daerah di Laporan Operasional dibagi dengan Total Pendapatan Laporan Operasional pada tahun yang bersangkutan.",
    "Fleksibilitas Keuangan": "Pendapatan Operasional Rutin pada Laporan Operasional dikurangi Beban Wajib kemudian hasilnya dibagi dengan Beban Wajib pada tahun yang bersangkutan.",
    "Kapasitas Layanan": "Total Aset Tetap (nilai buku) dibagi dengan jumlah penduduk suatu daerah pada tahun yang bersangkutan.",
    "Solvabilitas Anggaran": "Total Pendapatan pada Laporan Realisasi Anggaran dikurangi Pendapatan Dana Alokasi Khusus kemudian hasilnya dibagi Belanja Operasional tahun yang bersangkutan.",
    "Solvabilitas Operasional": "Total Pendapatan pada Laporan Operasional dikurangi beban operasional kemudian hasilnya dibagi beban operasional tahun yang bersangkutan.",
    "Komitmen Mempertahankan Layanan": "Beban pemeliharaan dibagi beban penyusutan aset tetap tahun yang bersangkutan.",
    "Komitmen Meningkatkan Layanan": "Belanja modal ditambah beban pemeliharaan kemudian hasilnya dibagi dengan beban penyusutan aset tetap tahun yang bersangkutan.",
    "Rasio Efektifitas Pengelolaan Pendapatan": "Pendapatan Asli Daerah dibagi PDRB suatu daerah tahun yang bersangkutan.",
    "Rasio Amanah": "Nilai absolut dari masing-masing selisih antara realisasi dengan anggaran pada pendapatan, belanja, penerimaan pembiayaan dan pengeluaran pembiayaan. Kemudian hasilnya dibagi dengan total anggaran belanja tahun yang bersangkutan."
}

# --- Judul dan Panduan Pengguna ---
st.title("🩺 Dashboard Diagnostik Prediksi Opini WTP")

with st.expander("Lihat Cara Membaca Hasil Analisis Ini"):
    st.write("""
        - **Prediksi & Keyakinan**: Menunjukkan prediksi opini (WTP/Tidak WTP) dan seberapa yakin model dengan prediksi tersebut.
        - **Analisis Faktor Pendorong**: Bagian terpenting. Ini menunjukkan **indikator mana yang paling berpengaruh** terhadap hasil prediksi.
            - **Tingkat Pengaruh**: Seberapa besar dampak sebuah indikator. Semakin panjang barnya, semakin besar pengaruhnya.
            - **Arah Pengaruh**:
                - **(+) Positif**: Nilai yang lebih tinggi pada indikator ini **mendukung** tercapainya opini WTP.
                - **(-) Negatif**: Nilai yang lebih tinggi pada indikator ini justru menjadi **penghambat**. Ini adalah area yang perlu menjadi fokus perbaikan.
    """)

# --- Antarmuka Input di Sidebar (VERSI PERBAIKAN) ---
with st.sidebar:
    st.header("Masukkan Indikator")
    input_data = {}
    
    # Loop untuk membuat input field dengan tooltip bawaan Streamlit
    for feature in features:
        input_data[feature] = st.number_input(
            label=feature, # Label akan muncul kembali
            help=tooltip_texts.get(feature, ""), # Gunakan parameter 'help' untuk tooltip
            step=0.01,
            format="%.4f"
        )
    
    st.markdown("---")
    predict_button = st.button("🚀 Lakukan Analisis Diagnostik", type="primary", use_container_width=True)

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
        'Koefisien LR': model_lr.coef_[0],
        'Nilai Input': input_df.iloc[0]
    })
    diagnostics_df['Arah Tanda'] = diagnostics_df['Koefisien LR'].apply(lambda x: '(+)' if x > 0 else '(-)')
    diagnostics_df['Arah Label'] = diagnostics_df['Koefisien LR'].apply(lambda x: 'Positif' if x > 0 else 'Negatif')
    diagnostics_df['Indikator Tampilan'] = diagnostics_df['Indikator'] + ' ' + diagnostics_df['Arah Tanda']
    diagnostics_df = diagnostics_df.sort_values('Tingkat Pengaruh', ascending=False).head(7)

    chart = alt.Chart(diagnostics_df).mark_bar().encode(
        x=alt.X('Tingkat Pengaruh:Q', title='Tingkat Pengaruh'),
        y=alt.Y('Indikator Tampilan:N', sort='-x', title='Indikator dan Arah Pengaruh'),
        color=alt.Color('Arah Label:N', scale=alt.Scale(domain=['Positif', 'Negatif'], range=['#2ECC71', '#E74C3C']), title="Arah Pengaruh"),
        tooltip=[alt.Tooltip('Indikator:N'), alt.Tooltip('Nilai Input:Q', format='.4f')]
    ).properties(title='Indikator Paling Berpengaruh Terhadap Prediksi')
    st.altair_chart(chart, use_container_width=True)

    st.header("Rekomendasi Kebijakan Otomatis")
    negative_drivers = diagnostics_df[diagnostics_df['Arah Label'] == 'Negatif']
    recommendations = []
    if not negative_drivers.empty:
        st.write("Berdasarkan analisis, berikut adalah area prioritas yang perlu mendapat perhatian:")
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
    st.info("Silakan masukkan nilai pada sidebar dan klik tombol 'Lakukan Analisis Diagnostik' untuk memulai.")
