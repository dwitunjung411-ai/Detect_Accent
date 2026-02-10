import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# 1. FIX KUSTOM OBJECT (MODIFIKASI RADIKAL)
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        # Buang semua argumen yang berpotensi menyebabkan error 'str'
        kwargs.pop('embedding_model', None)
        kwargs.pop('name', None) 
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return inputs

    @classmethod
    def from_config(cls, config):
        # Membersihkan config secara total dari atribut bermasalah
        for key in ["embedding_model", "name", "trainable", "dtype"]:
            config.pop(key, None)
        return cls(**config)

# ==========================================================
# 2. LOAD MODEL DENGAN ERROR HANDLING LEBIH LUAS
# ==========================================================
st.set_page_config(page_title="Deteksi Aksen Indonesia", page_icon="🎙️", layout="wide")

@st.cache_resource
def load_accent_model():
    model_path = "model_aksen.keras"
    if not os.path.exists(model_path):
        return None
    
    try:
        # Gunakan custom_objects yang sangat minimalis
        custom_objects = {
            'PrototypicalNetwork': PrototypicalNetwork,
            'embedding_model': tf.keras.layers.Layer # Definisikan sebagai Layer kosong
        }
        
        # Load dengan compile=False dan safe_mode=False
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects, 
            compile=False,
            safe_mode=False
        )
        return model
    except Exception as e:
        # Jika masih gagal, tampilkan instruksi debug
        st.error(f"❌ Masalah Arsitektur: {str(e)}")
        st.info("💡 Tip: Pastikan versi TensorFlow di requirements.txt sama dengan versi saat training.")
        return None
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
    try:
        # Load audio 16kHz
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Ekstraksi MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        # Reshape ke (1, 40)
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
# 4. UI UTAMA
# ==========================================================
def main():
    st.title("🎙️ Deteksi Aksen Indonesia")
    st.divider()

    model = load_accent_model()

    if model:
        st.sidebar.success("✅ Model Berhasil Dimuat")
    else:
        st.sidebar.error("❌ Model Gagal Dimuat")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📥 Input Audio")
        audio_file = st.file_uploader("Upload file (.wav, .mp3)", type=["wav", "mp3"])
        
        if audio_file:
            st.audio(audio_file)
            if st.button("🚀 Analisis Aksen", type="primary", use_container_width=True):
                if model:
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
                    st.error("Model tidak tersedia.")

if __name__ == "__main__":
    main()

