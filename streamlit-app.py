import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# DEFINISI CUSTOM LAYER / MODEL
# ==========================================================
# Berdasarkan error log, model Anda terdaftar sebagai 'Custom>PrototypicalNetwork'
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def call(self, inputs):
        # Ini adalah placeholder agar Keras bisa menginisialisasi class
        return inputs

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(page_title="Deteksi Aksen Indonesia", page_icon="🎙️", layout="wide")

# ==========================================================
# FUNGSI LOAD MODEL (FIX PROTOTYPICAL NETWORK)
# ==========================================================
@st.cache_resource
def load_accent_model():
    model_path = "model_aksen.keras"
    
    if not os.path.exists(model_path):
        st.error(f"❌ File '{model_path}' tidak ditemukan.")
        return None
    
    try:
        # Kita masukkan PrototypicalNetwork ke dalam custom_objects
        custom_objects = {
            'PrototypicalNetwork': PrototypicalNetwork,
            'embedding_model': lambda **kwargs: None # Antisipasi error sebelumnya
        }
        
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects, 
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"Detail Error: {str(e)}")
        return None

# ==========================================================
# FUNGSI PREDIKSI
# ==========================================================
def predict_accent(audio_path, model):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        # Ekstraksi MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        X = np.expand_dims(mfcc_mean, axis=0)
        
        pred = model.predict(X, verbose=0)
        
        classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        idx = np.argmax(pred[0])
        conf = pred[0][idx] * 100
        
        return classes[idx], conf
    except Exception as e:
        return f"Error Prediksi: {str(e)}", 0

# ==========================================================
# UI UTAMA
# ==========================================================
def main():
    st.title("🎙️ Deteksi Aksen Indonesia")
    st.divider()

    model = load_accent_model()
    
    if model:
        st.sidebar.success("✅ Model Prototypical siap digunakan")
    else:
        st.warning("⚠️ Gagal memuat arsitektur PrototypicalNetwork.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📥 Input Audio")
        audio_file = st.file_uploader("Upload (.wav, .mp3)", type=["wav", "mp3"])
        
        if audio_file:
            st.audio(audio_file)
            if st.button("🚀 Analisis Aksen", type="primary", use_container_width=True):
                with st.spinner("Menganalisis..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_file.getbuffer())
                        path = tmp.name
                    
                    label, conf = predict_accent(path, model)
                    
                    with col2:
                        st.subheader("📊 Hasil")
                        if conf > 0:
                            st.success(f"Aksen: **{label}** ({conf:.1f}%)")
                        else:
                            st.error(label)
                    os.unlink(path)

if __name__ == "__main__":
    main()
