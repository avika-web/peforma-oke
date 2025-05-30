import streamlit as st
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# --- Setup Fuzzy Variables ---
def build_fuzzy_system():
    ujian = ctrl.Antecedent(np.arange(0, 101, 1), 'ujian')
    tugas = ctrl.Antecedent(np.arange(0, 101, 1), 'tugas')
    kehadiran = ctrl.Antecedent(np.arange(0, 101, 1), 'kehadiran')
    partisipasi = ctrl.Antecedent(np.arange(0, 101, 1), 'partisipasi')
    proyek = ctrl.Antecedent(np.arange(0, 101, 1), 'proyek')
    kuis = ctrl.Antecedent(np.arange(0, 101, 1), 'kuis')

    performa = ctrl.Consequent(np.arange(0, 101, 1), 'performa')

    for var in [ujian, tugas, kehadiran, partisipasi, proyek, kuis]:
        var['rendah'] = fuzz.trapmf(var.universe, [0, 0, 40, 60])
        var['sedang'] = fuzz.trimf(var.universe, [50, 65, 80])
        var['tinggi'] = fuzz.trapmf(var.universe, [70, 85, 100, 100])

    performa['buruk'] = fuzz.trapmf(performa.universe, [0, 0, 40, 60])
    performa['cukup'] = fuzz.trimf(performa.universe, [50, 65, 80])
    performa['baik'] = fuzz.trapmf(performa.universe, [70, 85, 100, 100])

    rules = [
        ctrl.Rule(ujian['tinggi'] & tugas['tinggi'] & kehadiran['tinggi'] & partisipasi['tinggi'] & proyek['tinggi'] & kuis['tinggi'], performa['baik']),
        ctrl.Rule(ujian['sedang'] | tugas['sedang'], performa['cukup']),
        ctrl.Rule(ujian['rendah'] | kehadiran['rendah'], performa['buruk']),
    ]

    sistem = ctrl.ControlSystem(rules)
    simulasi = ctrl.ControlSystemSimulation(sistem)
    return simulasi

simulasi = build_fuzzy_system()

st.title("🎓 Sistem Prediksi Performa Akademik Mahasiswa")

st.sidebar.header("📌 Mode Input")
mode = st.sidebar.radio("Pilih Mode", ["Manual", "Upload Dataset"])

if mode == "Manual":
    st.subheader("🧑‍🎓 Input Nilai Mahasiswa")
    inputs = {}
    for label in ['ujian', 'tugas', 'kehadiran', 'partisipasi', 'proyek', 'kuis']:
        inputs[label] = st.slider(f"{label.capitalize()}", 0, 100, 60)

    for key, val in inputs.items():
        simulasi.input[key] = val

    simulasi.compute()
    nilai = simulasi.output['performa']

    if nilai <= 60:
        kategori = 'Buruk'
    elif nilai <= 80:
        kategori = 'Cukup'
    else:
        kategori = 'Baik'

    st.success(f"📈 Skor Performa: **{nilai:.2f}**\n\n📊 Kategori: **{kategori}**")

elif mode == "Upload Dataset":
    st.subheader("📄 Upload Dataset CSV")
    file = st.file_uploader("Pilih file CSV dengan kolom: ujian, tugas, kehadiran, partisipasi, proyek, kuis", type=['csv'])
    if file:
        df = pd.read_csv(file)
        hasil = []
        sim = build_fuzzy_system()

        for _, row in df.iterrows():
            try:
                for kol in ['ujian', 'tugas', 'kehadiran', 'partisipasi', 'proyek', 'kuis']:
                    sim.input[kol] = row[kol]
                sim.compute()
                skor = sim.output['performa']
                hasil.append(skor)
            except:
                hasil.append(np.nan)

        df['skor_performa'] = hasil
        df['kategori'] = pd.cut(df['skor_performa'], bins=[0,60,80,100], labels=['Buruk','Cukup','Baik'])
        st.dataframe(df)

        fig, ax = plt.subplots()
        df['kategori'].value_counts().plot(kind='bar', ax=ax, color=['red','orange','green'])
        ax.set_title("Distribusi Kategori Performa")
        st.pyplot(fig)

        st.download_button("📥 Download Hasil", df.to_csv(index=False), "hasil_prediksi.csv", "text/csv")
