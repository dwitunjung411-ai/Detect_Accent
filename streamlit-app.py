import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
import os

# ==========================================================
# 1. CLASS PROTOTYPICAL NETWORK (WAJIB ADA SAAT LOAD MODEL)
# ==========================================================
@tf.keras.utils.register_keras_serializable()
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None):
        super(PrototypicalNetwork, self).__init__()
        self.embedding = embedding_model

    def call(self, support_set, query_set, support_labels, n_way):
        # Embedding support & query
        support_embeddings = self.embedding(support_set)
        query_embeddings = self.embedding(query_set)

        prototypes = []
        for c in range(n_way):
            class_indices = tf.where(tf.equal(support_labels, c))
            class_embeddings = tf.gather_nd(support_embeddings, class_indices)
            class_prototype = tf.reduce_mean(class_embeddings, axis=0)
            prototypes.append(class_prototype)

        prototypes = tf.stack(prototypes)

        # Hitung jarak
        distances = tf.norm(
            tf.expand_dims(query_embeddings, 1) - prototypes,
            axis=2
        )

        return -distances


# ==========================================================
# 2. LOAD MODEL
# ==========================================================
@st.cache_resource
def load_protonet_model():
    model = load_model(
        "model_aksen.keras",
        custom_objects={"PrototypicalNetwork": PrototypicalNetwork}
    )
    return model

model = load_protonet_model()

# ==========================================================
# 3. LABEL AKSEN
# ==========================================================
label_map = {
    0: "Betawi",
    1: "Sunda",
    2: "Jawa_Tengah",
    3: "Jawa_Timur",
    4: "Yogyakarta"
}

n_way = len(label_map)

# ==========================================================
# 4. EXTRACT MFCC
# ==========================================================
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean


# ==========================================================
# 5. STREAMLIT UI
# ==========================================================
st.title("🎙️ Deteksi Aksen (Prototypical Network)")

uploaded_file = st.file_uploader("Upload Audio", type=["wav"])

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if st.button("🔍 Detect Accent"):

        try:
            # ==================================================
            # EXTRACT QUERY SET
            # ==================================================
            query_features = extract_features(tmp_path)

            if query_features is None:
                st.error("Feature extraction gagal.")
                st.stop()

            query_tensor = np.expand_dims(query_features, axis=0)
            query_tensor = tf.convert_to_tensor(query_tensor, dtype=tf.float32)

            # ==================================================
            # DUMMY SUPPORT SET
            # (HARUS SAMA DIMENSI DENGAN TRAINING)
            # ==================================================
            support_set = np.random.rand(n_way * 5, 40)
            support_labels = np.repeat(np.arange(n_way), 5)

            support_tensor = tf.convert_to_tensor(support_set, dtype=tf.float32)
            support_labels_tensor = tf.convert_to_tensor(support_labels, dtype=tf.int32)

            # ==================================================
            # PANGGIL MODEL (TIDAK PAKAI .call())
            # ==================================================
            logits = model(
                support_tensor,
                query_tensor,
                support_labels_tensor,
                n_way
            )

            predicted_class = tf.argmax(logits, axis=1).numpy()[0]
            predicted_label = label_map[int(predicted_class)]

            st.success(f"🎯 Predicted Accent: {predicted_label}")

        except Exception as e:
            st.error(f"Terjadi error: {str(e)}")

    os.remove(tmp_path)
