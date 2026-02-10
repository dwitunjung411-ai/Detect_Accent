import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tempfile
import os

# =========================================================
# KONFIGURASI
# =========================================================
SAMPLE_RATE = 16000
MAX_LEN = 5        # detik
N_MFCC = 40
N_SUPPORT = 5

AUDIO_DIR = "data/audio"   # SESUAIKAN DENGAN FOLDER AUDIO

# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================
@st.cache_resource
def load_embedding_model():
    model = tf.keras.models.load_model(
        "model_aksen.keras",  # NAMA MODEL KAMU
        compile=False
    )
    return model

embedding_model = load_embedding_model()

# =========================================================
# LOAD METADATA (SESUAI CSV KAMU)
# =========================================================
@st.cache_data
def load_metadata():
    df = pd.read_csv("metadata.csv")
    return df

metadata = load_metadata()

# =========================================================
# KOLOM METADATA (FIXED SESUAI FILE KAMU)
# =========================================================
AKSEN_COL = "label_aksen"
PATH_COL = "file_name"

# =========================================================
# LABEL MAP (HARUS SAMA DENGAN TRAINING)
# =========================================================
label_map = {
    "Betawi": 0,
    "Sunda": 1,
    "Jawa_Tengah": 2,
    "Jawa_Timur": 3
}

id_to_label = {v: k for k, v in label_map.items()}

metadata["label_id"] = metadata[AKSEN_COL].map(label_map)

# gabungkan folder + nama file
metadata["file_path"] = metadata[PATH_COL].apply(
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
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    return mfcc.T  # (time, mfcc)

# =========================================================
# AUTO SUPPORT SET DARI METADATA
# =========================================================
def generate_support_set(metadata):
    support_x, support_y = [], []

    for aksen, label_id in label_map.items():
        samples = metadata[metadata["label_id"] == label_id]

        if len(samples) == 0:
            continue

        samples = samples.sample(
            n=min(N_SUPPORT, len(samples)),
            random_state=42
        )

        for path in samples["file_path"]:
            if os.path.exists(path):
                feat = extract_feature(path)
                support_x.append(feat)
                support_y.append(label_id)

    return np.array(support_x), np.array(support_y)

# =========================================================
# COMPUTE PROTOTYPES
# =========================================================
def compute_prototypes(model, support_x, support_y):
    embeddings = model.predict(support_x, verbose=0)

    prototypes, labels = [], np.unique(support_y)

    for lbl in labels:
        proto = embeddings[support_y == lbl].mean(axis=0)
        prototypes.append(proto)

    return np.array(prototypes), labels

# =========================================================
# PREDICTION
# =========================================================
def predict_accent(model, query_feat, prototypes, labels):
    query_emb = model.predict(
        np.expand_dims(query_feat, axis=0),
        verbose=0
    )

    distances = tf.norm(prototypes - query_emb, axis=1)
    pred_idx = np.argmin(distances)

    return labels[pred_idx], distances.numpy()

# =========================================================
# STREAMLIT UI
# =========================================================
st.title("🎙️ Deteksi Aksen Bahasa (Few-Shot Learning)")
st.write("Prototypical Network + Metadata Otomatis")

uploaded_file = st.file_uploader(
    "Upload audio (.wav)",
    type=["wav"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        query_path = tmp.name

    st.audio(uploaded_file)

    with st.spinner("🔍 Memproses audio..."):
        # 1. Support set
        support_x, support_y = generate_support_set(metadata)

        if len(support_x) == 0:
            st.error("❌ Support set kosong. Cek metadata & folder audio.")
            st.stop()

        # 2. Prototype
        prototypes, proto_labels = compute_prototypes(
            embedding_model,
            support_x,
            support_y
        )

        # 3. Query feature
        query_feat = extract_feature(query_path)

        # 4. Predict
        pred_label, distances = predict_accent(
            embedding_model,
            query_feat,
            prototypes,
            proto_labels
        )

    st.success(f"✅ Aksen terdeteksi: **{id_to_label[pred_label]}**")

    # Confidence
    confidence = tf.nn.softmax(-distances).numpy()

    st.subheader("Confidence Score")
    for i, lbl in enumerate(proto_labels):
        st.write(f"{id_to_label[lbl]} : {confidence[i]*100:.2f}%")

    os.remove(query_path)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("Few-Shot Learning | Prototypical Network | Skripsi")
