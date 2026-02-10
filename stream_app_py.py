import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==========================================================
# 1. DEFINISI CLASS PROTOTYPICAL NETWORK
# ==========================================================
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

# ==========================================================
# 2. FUNGSI LOAD DATA (MODEL & METADATA)
# ==========================================================
@st.cache_resource
def load_accent_model():
    print("Sedang mencoba memuat model...") # Ini akan muncul di log Manage App
    # ... kode load model kamu ...
    print("Model berhasil dimuat ke memori!")
    return model

@st.cache_data
def load_metadata_df():
    csv_path = "metadata.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

# ==========================================================
# 3. FUNGSI PREDIKSI
# ==========================================================
def predict_accent(audio_path, model):
    if model is None: return "Model tidak tersedia"
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        input_data = np.expand_dims(mfcc_scaled, axis=0)

        prediction = model.predict(input_data)
        aksen_classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        return aksen_classes[np.argmax(prediction)]
    except Exception as e:
        return f"Error Analisis: {str(e)}"

# ==========================================================
# 4. MAIN UI (PENGATURAN LEBAR & PEMBERSIHAN)
# ==========================================================
def main():
    # Menambahkan layout="wide" untuk memperlebar tampilan
    st.set_page_config(page_title="Deteksi Aksen Prototypical", page_icon="🎙️", layout="wide")

    model_aksen = load_accent_model()
    df_metadata = load_metadata_df()

    st.title("🎙️ Accent Recognation")
    st.divider()

    with st.sidebar:
        st.header("⚙️ Status Sistem")
        if model_aksen:
            st.success("Model: Online")
        else:
            st.error("Model: Offline")

    # Mengatur perbandingan kolom (misal 1:1.2 agar kolom hasil lebih lega)
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📤 Input Audio")
        audio_file = st.file_uploader("Upload file (.wav, .mp3)", type=["wav", "mp3"])

        if audio_file:
            st.audio(audio_file)
            if st.button("🚀 Extract Feature and  Detect", type="primary", use_container_width=True):
                if model_aksen:
                    with st.spinner("Sedang memproses..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(audio_file.getbuffer())
                            tmp_path = tmp.name

                        # Jalankan fungsi prediksi
                        hasil_aksen = predict_accent(tmp_path, model_aksen)

                        # Pencarian metadata berdasarkan nama file
                        user_info = None
                        if df_metadata is not None:
                            match = df_metadata[df_metadata['file_name'] == audio_file.name]
                            if not match.empty:
                                user_info = match.iloc[0].to_dict()

                        with col2:
                            st.subheader("📊 Hasil Analisis")
                            # Box hasil aksen
                            st.info(f"### Aksen Terdeteksi: **{hasil_aksen}**")

                            st.write("---")
                            st.subheader("🔹Info Pembicara")
                            if user_info:
                                st.write(f"📅Usia: {user_info.get('usia', '-')}")
                                st.write(f"🗣️Gender: {user_info.get('gender', '-')}")
                                st.write(f"📍Provinsi: {user_info.get('provinsi', '-')}")
                            else:
                                st.warning("Data file ini tidak terdaftar di metadata.csv")

                        # Hapus temporary file
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                else:
                    st.error("Model gagal dimuat. Cek log server.")

if __name__ == "__main__":
    main()
