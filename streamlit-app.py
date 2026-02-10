import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# 1. FIX REGISTRASI CUSTOM MODEL
# ==========================================================
# Kita gunakan decorator untuk meregistrasi class agar Keras mengenalinya
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        # Mengabaikan argumen kustom yang menyebabkan error load
        kwargs.pop('embedding_model', None)
        super().__init__(*args, **kwargs)
    
    def call(self, inputs):
        return inputs

    @classmethod
    def from_config(cls, config):
        # Menghapus config yang tidak dikenal sebelum inisialisasi
        config.pop('embedding_model', None)
        return cls(**config)

# ==========================================================
# 2. KONFIGURASI UI
# ==========================================================
st.set_page_config(page_title="Deteksi Aksen Indonesia", page_icon="🎙️", layout="wide")

@st.cache_resource
def load_accent_model():
    model_path = "model_aksen.keras"
    if not os.path.exists(model_path):
        return None
    
    try:
        # Mencoba load dengan berbagai kombinasi custom objects
        custom_objects = {
            'PrototypicalNetwork': PrototypicalNetwork,
            'embedding_model': lambda **kwargs: None
        }
        
        # Load dengan compile=False agar tidak mengecek optimizer/loss kustom
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects, 
            compile=False,
            safe_mode=False  # Sangat penting untuk model kustom
        )
        return model
    except Exception as e:
        st.error(f"Gagal memuat arsitektur: {str(e)[:200]}")
        return None

# ==========================================================
# 3. FUNGSI PREDIKSI
# ==========================================================
def predict_accent(audio_path, model):
    try:
        # Load audio 16kHz
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Ekstraksi Feature MFCC (Samakan dengan spek saat training)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        # Sesuaikan dimensi input (Batch, Features)
        X = np.expand_dims(mfcc_mean, axis=0)
        
        # Prediksi
        pred = model.predict(X, verbose=0)
        
        classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        idx = np.argmax(pred[0])
        conf = pred[0][idx] * 100
        
        return classes[idx], conf
    except Exception as e:
        return f"Error Prediksi: {str(e)}", 0

# ==========================================================
# 4. MAIN INTERFACE
# ==========================================================
def main():
    st.title("🎙️ Deteksi Aksen Indonesia")
    st.divider()

    model = load_accent_model()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📥 Input Audio")
        audio_file = st.file_uploader("Upload (.wav, .mp3)", type=["wav", "mp3"])
        
        if audio_file:
            st.audio(audio_file)
            if st.button("🚀 Analisis Aksen", type="primary", use_container_width=True):
                if model is not None:
                    with st.spinner("Menganalisis..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(audio_file.getbuffer())
                            path = tmp.name
                        
                        label, conf = predict_accent(path, model)
                        
                        with col2:
                            st.subheader("📊 Hasil")
                            if conf > 0:
                                st.success(f"Aksen Terdeteksi: **{label}**")
                                st.info(f"Keyakinan: **{conf:.1f}%**")
                            else:
                                st.error(label)
                        
                        os.unlink(path)
                else:
                    st.error("Model gagal dimuat. Periksa log di atas.")

if __name__ == "__main__":
    main()
