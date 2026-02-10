import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ===============================
# KONFIGURASI
# ===============================
SAMPLE_RATE = 16000
DURATION = 3
N_MFCC = 40

MODEL_PATHS = {
    "age": "models/model_aksen.keras",
    "gender": "models/model_aksen.keras",
    "province": "models/model_aksen.keras",
    "accent": "models/model_aksen.keras",
}

# ===============================
# LOAD METADATA (AUTOMATIS)
# ===============================
@st.cache_data
def load_metadata():
    if not os.path.exists("metadata.csv"):
        return None
    return pd.read_csv("metadata.csv")

metadata = load_metadata()

# ===============================
# LOAD MODEL (CACHED)
# ===============================
@st.cache_resource
def load_model_safe(path):
    if os.path.exists(path):
        return tf.keras.models.load_model(path, compile=False)
    return None

models = {k: load_model_safe(v) for k, v in MODEL_PATHS.items()}

# ===============================
# AUDIO PREPROCESSING
# ===============================
def extract_mfcc(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc.reshape(1, -1)

# ===============================
# LABEL OTOMATIS DARI METADATA
# ===============================
def get_unique_labels(column):
    if metadata is None or column not in metadata.columns:
        return None
    return sorted(metadata[column].dropna().unique().tolist())

LABELS = {
    "gender": get_unique_labels("gender"),
    "province": get_unique_labels("province"),
    "accent": get_unique_labels("accent"),
}

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Multitask Speech Analyzer", layout="centered")
st.title("🎙️ Multitask Speech Analyzer")
st.write("Deteksi **Usia, Gender, Provinsi, dan Aksen** dari satu audio")

uploaded_file = st.file_uploader(
    "Upload audio (.wav / .mp3)",
    type=["wav", "mp3"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(uploaded_file)

    with st.spinner("🔍 Memproses audio..."):
        features = extract_mfcc(audio_path)

    st.subheader("📊 Hasil Prediksi")

    col1, col2 = st.columns(2)

    # ===============================
    # USIA (REGRESI)
    # ===============================
    if models["age"]:
        age_pred = models["age"].predict(features)[0][0]
        col1.metric("🧓 Usia (tahun)", f"{int(age_pred)}")

    # ===============================
    # GENDER
    # ===============================
    if models["gender"] and LABELS["gender"]:
        gender_pred = models["gender"].predict(features)
        gender_label = LABELS["gender"][np.argmax(gender_pred)]
        col2.metric("🚻 Gender", gender_label)

    # ===============================
    # PROVINSI
    # ===============================
    if models["province"] and LABELS["province"]:
        prov_pred = models["province"].predict(features)
        prov_label = LABELS["province"][np.argmax(prov_pred)]
        col1.metric("📍 Provinsi", prov_label)

    # ===============================
    # AKSEN
    # ===============================
    if models["accent"] and LABELS["accent"]:
        acc_pred = models["accent"].predict(features)
        acc_label = LABELS["accent"][np.argmax(acc_pred)]
        col2.metric("🗣️ Aksen", acc_label)

    os.remove(audio_path)

# ===============================
# DEBUG INFO
# ===============================
with st.expander("🔧 Debug Info"):
    st.write("Metadata Loaded:", metadata is not None)
    st.write("Available Labels:", LABELS)
    st.write("Loaded Models:", {k: v is not None for k, v in models.items()})
