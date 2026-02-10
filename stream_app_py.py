import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tempfile
import os

# =========================================================
# KONFIGURASI AUDIO
# =========================================================
SAMPLE_RATE = 16000
MAX_LEN = 5
N_MFCC = 40
N_SUPPORT = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# AUTO DETECT AUDIO DIRECTORY
# =========================================================
CANDIDATE_AUDIO_DIRS = [
    os.path.join(BASE_DIR, "audio"),
    os.path.join(BASE_DIR, "data", "audio"),
    os.path.join(BASE_DIR, "data"),
    BASE_DIR
]

AUDIO_DIR = None
for d in CANDIDATE_AUDIO_DIRS:
    if os.path.exists(d) and any(f.endswith(".wav") for f in os.listdir(d)):
        AUDIO_DIR = d
        break

if AUDIO_DIR is None:
    st.error("❌ Folder audio tidak ditemukan otomatis.")
    st.stop()

st.success(f"📁 Audio folder terdeteksi: `{AUDIO_DIR}`")

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        os.path.join(BASE_DIR, "model_aksen.keras"),
        compile=False
    )

model = load_model()

# =========================================================
# LOAD METADATA
# =========================================================
@st.cache_data
def load_metadata():
    return pd.read_csv(os.path.join(BASE_DIR, "metadata.csv"))

metadata = load_metadata()

# =========================================================
# METADATA CONFIG (SESUAI FILE KAMU)
# =========================================================
AKSEN_COL = "label_aksen"
FILENAME_COL = "file_name"

label_map = {
    "Betawi": 0,
    "Sunda": 1,
    "Jawa_Tengah": 2,
    "Jawa_Timur": 3,
    "YogyaKarta": 4
}

id_to_label = {v: k for k, v in label_map.items()}

metadata["label_id"] = metadata[AKSEN_COL].map(label_map)
metadata["file_path"] = metadata[FILENAME_COL].apply(
    lambda x: os.path.join(AUDIO_DIR, x)
)

# =========================================================
# FEATURE EXTRACTION
# =========================================================
def extract_feature(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE)
    audio = audio[:SAMPLE_RATE * MAX_LEN]
    if len(audio) < SAMPLE_RATE * MAX_LEN:
        audio = np.pad(audio, (0, SAMPLE_RATE * MAX_LEN - len(audio)))

    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC
    )
    return mfcc.T

# =========================================================
# SUPPORT SET
# =========================================================
def generate_support_set(df):
    support_x, support_y = [], []

    for aksen, label_id in label_map.items():
        samples = df[
            (df["label_id"] == label_id) &
            (df["file_path"].apply(os.path.exists))
        ]

        if samples.empty:
            continue

        samples = samples.sample(
            n=min(N_SUPPORT, len(samples)),
            random_state=42
        )

        for path in samples["file_path"]:
            support_x.append(extract_feature(path))
            support_y.append(label_id)

    return np.array(support_x), np.array(support_y)

# =========================================================
# PROTOTYPE
# =========================================================
def compute_prototypes(model, x, y):
    emb = model.predict(x, verbose=0)
    protos, labels = [], np.unique(y)

    for lbl in labels:
        protos.append(emb[y == lbl].mean(axis=0))

    return np.array(protos), labels

# =========================================================
# PREDICT
# =========================================================
def predict(model, feat, protos, labels):
    q = model.predict(np.expand_dims(feat, 0), verbose=0)
    dist = tf.norm(protos - q, axis=1)
    return labels[np.argmin(dist)], dist.numpy()

# =========================================================
# STREAMLIT UI
# =========================================================
st.title("🎙️ Deteksi Aksen Bahasa (Few-Shot Learning)")
st.caption("Prototypical Network • Support otomatis dari metadata")

uploaded = st.file_uploader("Upload audio (.wav)", type=["wav"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded.read())
        query_path = tmp.name

    st.audio(uploaded)

    with st.spinner("🔍 Memproses..."):
        sx, sy = generate_support_set(metadata)

        if len(sx) == 0:
            st.error("❌ Support set kosong. Cek metadata & nama file.")
            st.stop()

        protos, labels = compute_prototypes(model, sx, sy)
        feat = extract_feature(query_path)
        pred, dist = predict(model, feat, protos, labels)

    st.success(f"✅ Aksen terdeteksi: **{id_to_label[pred]}**")

    conf = tf.nn.softmax(-dist).numpy()
    st.subheader("Confidence")
    for i, lbl in enumerate(labels):
        st.write(f"{id_to_label[lbl]} : {conf[i]*100:.2f}%")

    os.remove(query_path)

st.markdown("---")
st.caption("Skripsi • Few-Shot Learning • Prototypical Network")
