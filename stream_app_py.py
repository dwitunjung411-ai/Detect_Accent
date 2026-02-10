import streamlit as st
import tensorflow as tf
import os
import numpy as np

# 1. DEFINISI CUSTOM CLASS (Wajib sama dengan saat training)
# Harus didaftarkan agar Keras bisa mengenali layer/model custom
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, support_set, query_set, support_labels, n_way):
        return self.embedding(query_set)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embedding_model": tf.keras.layers.serialize(self.embedding)
        })
        return config

# 2. FUNGSI LOAD MODEL DENGAN ERROR HANDLING
@st.cache_resource
def load_accent_model():
    # Gunakan path relatif sederhana untuk Streamlit Cloud
    model_name = "model_aksen.keras"
    
    if not os.path.exists(model_name):
        st.error(f"File {model_name} tidak ditemukan di root directory!")
        return None

    try:
        # Menyiapkan custom objects untuk Few-Shot Learning
        custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
        model = tf.keras.models.load_model(model_name, custom_objects=custom_objects)
        return model
    except Exception as e:
        # Menampilkan error spesifik di UI jika loading gagal
        st.error(f"Gagal memuat model: {str(e)}")
        return None

# 3. SETTING PAGE
st.set_page_config(page_title="Accent Recognition", layout="wide")

# 4. SIDEBAR STATUS
st.sidebar.title("⚙️ Status Sistem")
model = load_accent_model()

if model is not None:
    st.sidebar.success("Model: Online")
else:
    st.sidebar.error("Model: Offline")

# 5. MAIN UI
st.title("🎙️ Accent Recognition")
st.write("Aplikasi pendeteksi aksen regional menggunakan Multitask CNN & Few-Shot Learning.")

st.divider()

# Input Audio
st.subheader("📤 Input Audio")
uploaded_file = st.file_uploader("Upload file (.wav, .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Mulai Prediksi"):
        if model is not None:
            with st.spinner('Menganalisis aksen...'):
                # Tambahkan logika preprocessing audio & prediksi di sini
                st.info("Fitur prediksi sedang disiapkan.")
        else:
            st.error("Sistem tidak siap. Periksa status model di sidebar.")
