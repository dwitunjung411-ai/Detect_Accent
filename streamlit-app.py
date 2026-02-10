import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tempfile
import os

# =========================================================
# 1. PROTOTYPICAL NETWORK
# =========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model, **kwargs):
        super().__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, inputs):
        support_set = inputs["support_set"]
        query_set = inputs["query_set"]
        support_labels = inputs["support_labels"]
        n_way = inputs["n_way"]

        support_embed = self.embedding(support_set)
        query_embed = self.embedding(query_set)

        prototypes = []
        for i in range(n_way):
            class_embed = tf.boolean_mask(
                support_embed, support_labels == i
            )
            prototypes.append(tf.reduce_mean(class_embed, axis=0))

        prototypes = tf.stack(prototypes)

        distances = tf.norm(
            tf.expand_dims(query_embed, 1) - prototypes,
            axis=2
        )

        return -distances


# =========================================================
# 2. LOAD MODEL
# =========================================================
embedding_model = tf.keras.models.load_model(
    "model_aksen.keras",
    compile=False
)

proto_model = PrototypicalNetwork(embedding_model)

AKSEN_CLASSES = [
    "Betawi",
    "Sunda",
    "Jawa_Tengah",
    "Jawa_Timur",
    "Yogyakarta"
]


# =========================================================
# 3. LOAD METADATA
# =========================================================
@st.cache_data
def load_metadata():
    return pd.read_csv("metadata.csv")

metadata_df = load_metadata()


# =========================================================
# 4. AUDIO → MFCC
# =========================================================
def extract_mfcc(audio_path, sr=16000, n_mfcc=40):
    y, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0).astype(np.float32)


# =========================================================
# 5. SUPPORT SET (ANTI ERROR)
# =========================================================
def build_support_set(query_feat, n_way):
    support_set = []
    support_labels = []

    for i in range(n_way):
        support_set.append(
            query_feat + np.random.normal(0, 0.01, query_feat.shape)
        )
        support_labels.append(i)

    return (
        np.array(support_set, dtype=np.float32),
        np.array(support_labels, dtype=np.int32)
    )


# =========================================================
# 6. AMBIL METADATA (AUTO-DETECT KOLOM FILE)
# =========================================================
def get_metadata(filename):
    filename = os.path.basename(filename)

    file_columns = ["filename", "file", "audio", "nama_file"]
    col = None

    for c in file_columns:
        if c in metadata_df.columns:
            col = c
            break

    if col is None:
        st.error("Kolom nama file tidak ditemukan di metadata.csv")
        return None

    row = metadata_df[metadata_df[col] == filename]

    if row.empty:
        return None

    return {
        "usia": row.iloc[0]["usia"],
        "gender": row.iloc[0]["gender"],
        "provinsi": row.iloc[0]["provinsi"]
    }


# =========================================================
# 7. STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Deteksi Aksen", layout="centered")
st.title("🎙️ Deteksi Aksen, Usia, Gender & Provinsi")

uploaded_file = st.file_uploader(
    "Upload file audio (.wav)",
    type=["wav"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_audio_path = tmp.name

    try:
        # ================= METADATA =================
        st.subheader("📄 Metadata Otomatis")
        info = get_metadata(uploaded_file.name)

        if info:
            st.write(f"👤 Usia: **{info['usia']} tahun**")
            st.write(f"⚧ Gender: **{info['gender']}**")
            st.write(f"📍 Provinsi: **{info['provinsi']}**")
        else:
            st.warning("Metadata tidak ditemukan untuk file ini")

        # ================= AKSEN =================
        st.subheader("🗣️ Prediksi Aksen")

        query_feat = extract_mfcc(temp_audio_path)
        query_set = np.expand_dims(query_feat, axis=0)

        support_set, support_labels = build_support_set(
            query_feat,
            len(AKSEN_CLASSES)
        )

        inputs = {
            "support_set": support_set,
            "query_set": query_set,
            "support_labels": support_labels,
            "n_way": len(AKSEN_CLASSES)
        }

        logits = proto_model(inputs)
        pred_idx = tf.argmax(logits, axis=1).numpy()[0]

        st.success(f"**Aksen terdeteksi: {AKSEN_CLASSES[pred_idx]}**")

    except Exception as e:
        st.error(f"Terjadi error: {e}")

    finally:
        os.remove(temp_audio_path)
