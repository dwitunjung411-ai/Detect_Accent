import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf
from tensorflow import keras

# ==========================================================
# DEFINISI CUSTOM CLASS
# ==========================================================
@keras.saving.register_keras_serializable(package="Custom")
class PrototypicalNetwork(keras.layers.Layer):
    """Custom layer untuk Prototypical Network"""
    
    def __init__(self, num_classes=5, embedding_dim=128, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
    
    def build(self, input_shape):
        super(PrototypicalNetwork, self).build(input_shape)
    
    def call(self, inputs, training=None):
        return inputs
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(PrototypicalNetwork, self).get_config()
        config.update({
            'num_classes': self.num_classes,
            'embedding_dim': self.embedding_dim
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ==========================================================
# CLEAR CACHE
# ==========================================================
st.cache_resource.clear()
st.cache_data.clear()

# ==========================================================
# LOAD MODEL
# ==========================================================
@st.cache_resource(show_spinner=False)
def load_model():
    # Cari file model otomatis
    possible_names = ["model_aksen.keras"]
    model_path = None
    
    for name in possible_names:
        if os.path.exists(name):
            model_path = name
            break
    
    if model_path is None:
        st.sidebar.error("❌ File model tidak ditemukan")
        st.sidebar.write("📁 File yang dicari:")
        for name in possible_names:
            st.sidebar.write(f"  - {name}")
        
        st.sidebar.divider()
        st.sidebar.write("📂 File di folder saat ini:")
        all_files = os.listdir(".")
        model_files = [f for f in all_files if f.endswith(('.keras', '.h5'))]
        if model_files:
            for f in model_files:
                st.sidebar.write(f"  ✓ {f}")
        else:
            st.sidebar.write("  (tidak ada file .keras atau .h5)")
        
        return None
    
    try:
        custom_objects = {'PrototypicalNetwork': PrototypicalNetwork}
        model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        st.sidebar.success(f"✅ Model loaded: {model_path}")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model:")
        st.sidebar.code(str(e)[:200])
        return None

# ==========================================================
# LOAD METADATA
# ==========================================================
@st.cache_data
def load_metadata():
    try:
        if os.path.exists("metadata.csv"):
            df = pd.read_csv("metadata.csv")
            df = df.dropna(subset=['file_name'])
            df['file_name'] = df['file_name'].str.strip()
            return df
    except Exception as e:
        st.sidebar.warning(f"⚠️ Metadata error: {str(e)[:100]}")
    return None

# ==========================================================
# PREDIKSI
# ==========================================================
def predict_accent(audio_path, model):
    if model is None:
        return "Model tidak tersedia", None
    
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        X = np.expand_dims(mfcc_mean, axis=0)
        
        pred = model.predict(X, verbose=0)
        
        classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        idx = np.argmax(pred[0])
        conf = pred[0][idx] * 100
        
        result = f"{classes[idx]} ({conf:.1f}%)"
        probs = {classes[i]: float(pred[0][i] * 100) for i in range(len(classes))}
        
        return result, probs
        
    except Exception as e:
        return f"Error: {str(e)}", None

# ==========================================================
# UI
# ==========================================================
st.set_page_config(page_title="Deteksi Aksen", page_icon="🎙️", layout="wide")

st.title("🎙️ Deteksi Aksen Indonesia")
st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan")
    if st.button("🔄 Clear Cache & Reload", use_container_width=True):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    st.divider()

# Load resources
model = load_model()
metadata = load_metadata()

if metadata is not None:
    st.sidebar.info(f"📊 {len(metadata)} metadata loaded")

# Main layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Input Audio")
    
    audio = st.file_uploader("Upload (.wav, .mp3)", type=["wav", "mp3"])
    
    if audio:
        st.audio(audio)
        
        if st.button("🚀 Analisis", type="primary", use_container_width=True):
            if model is None:
                st.error("❌ Model tidak tersedia. Cek sidebar untuk info detail.")
            else:
                with st.spinner("Menganalisis audio..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                        f.write(audio.getbuffer())
                        temp_path = f.name
                    
                    result, probs = predict_accent(temp_path, model)
                    
                    with col2:
                        st.subheader("📊 Hasil Prediksi")
                        
                        if "Error" in result:
                            st.error(result)
                        else:
                            st.success(f"**{result}**")
                            
                            if probs:
                                st.write("**Detail Confidence:**")
                                for name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                                    st.progress(prob/100, text=f"{name}: {prob:.1f}%")
                        
                        st.divider()
                        st.subheader("📋 Info Speaker")
                        
                        if metadata is not None:
                            match = metadata[metadata['file_name'] == audio.name]
                            if not match.empty:
                                info = match.iloc[0]
                                
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.metric("Usia", f"{info['usia']} Tahun")
                                    st.metric("Gender", info['gender'])
                                with c2:
                                    st.metric("Provinsi", info['provinsi'])
                                    st.metric("Label Aksen", info['label_aksen'])
                            else:
                                st.info("ℹ️ Data tidak ditemukan untuk file ini")
                        else:
                            st.info("ℹ️ Metadata tidak tersedia")
                    
                    try:
                        os.unlink(temp_path)
                    except:
                        pass

with col2:
    if not audio:
        st.info("👆 Upload file audio di sebelah kiri untuk memulai")

st.divider()
st.caption("🎯 Sistem Deteksi Aksen Bahasa Indonesia | Powered by Deep Learning")

