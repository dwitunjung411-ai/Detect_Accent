import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tempfile
import os
import zipfile
import shutil

# =========================================================
# KONFIGURASI
# =========================================================
SAMPLE_RATE = 16000
MAX_LEN = 5
N_MFCC = 40
N_SUPPORT = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_AUDIO_DIR = os.path.join(BASE_DIR, "_audio_tmp")

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
# LABEL MAP (SESUAI DATA KAMU)
# =========================================================
label_map = {
    "Betawi": 0,
    "Sunda": 1,
    "Jawa_Tengah": 2,
    "Jawa_Timur": 3,
    "YogyaKarta": 4
}
id_to_label = {v: k for k, v in label_map.items()}

metadata["label_id"] = metadata["label_aksen"].map(label_map)

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
def generate_support_set(df, audio_dir):
    support_x, support_y = [], []

    for aksen, label_id in label_map.items():
        samples = df[df["label_id"] == label_id]

        if samples.empty:
            continue

        samples = samples.sample(
            n=min(N_SUPPORT, len(samples)),
            random_state=42
        )

        for fname in samples["file_name"]:
            path = os.path.join(audio_dir, fname)
            if os.path.exists(path):
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
st.write("Support set dari metadata + audio ZIP (anti gagal deployment)")

zip_file = st.file_uploader(
    "Upload ZIP berisi audio support set (.wav)",
    type=["zip"]
)

query_file = st.file_uploader(
    "Upload audio query (.wav)",
    type=["wav"]
)

if zip_file and query_file:
    # bersihkan folder temp
    if os.path.exists(TMP_AUDIO_DIR):
        shutil.rmtree(TMP_AUDIO_DIR)
    os.makedirs(TMP_AUDIO_DIR)

    # extract zip
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(TMP_AUDIO_DIR)

    st.success("✅ Audio support set berhasil dimuat")

    # simpan query
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(query_file.read())
        query_path = tmp.name

    st.audio(query_file)

    with st.spinner("🔍 Memproses audio..."):
        sx, sy = generate_support_set(metadata, TMP_AUDIO_DIR)

        if len(sx) == 0:
            st.error("❌ Support set kosong. Nama file ZIP harus sama dengan metadata.")
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
st.caption("Few-Shot Learning • Prototypical Network • Skripsi")
