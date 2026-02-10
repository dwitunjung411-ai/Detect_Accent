import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# CLEAR CACHE
# ==========================================================
st.cache_resource.clear()
st.cache_data.clear()

# ==========================================================
# LOAD MODEL - VERSI PALING SIMPLE
# ==========================================================
@st.cache_resource(show_spinner=False)
def load_model_simple():
    model_path = "model_aksen.keras"
    
    if not os.path.exists(model_path):
        st.sidebar.error(f"❌ File tidak ditemukan: {model_path}")
        return None
    
    try:
        # Load dengan safe_mode False dan skip custom objects
        with tf.keras.utils.custom_object_scope({}):
            model = tf.keras.models.load_model(
                model_path,
                compile=False,
                safe_mode=False
            )
        st.sidebar.success("✅ Model berhasil dimuat")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)[:100]}")
        
        # Coba metode alternatif: load_weights only
        try:
            st.sidebar.info("🔄 Mencoba metode alternatif...")
            
            # Buat arsitektur sederhana
            inputs = tf.keras.Input(shape=(40,))
            x = tf.keras.layers.Dense(128, activation='relu')(inputs)
            x = tf.keras.layers.Dropout(0.3)(x)
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            # Load weights
            model.load_weights(model_path)
            
            st.sidebar.success("✅ Model dimuat (weights only)")
            return model
            
        except Exception as e2:
            st.sidebar.error(f"❌ Semua metode gagal")
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
    except:
        return None
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

# Load
model = load_model_simple()
metadata = load_metadata()

if metadata is not None:
    st.sidebar.info(f"📊 {len(metadata)} metadata loaded")

# Main
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Input Audio")
    audio = st.file_uploader("Upload (.wav, .mp3)", type=["wav", "mp3"])
    
    if audio:
        st.audio(audio)
        
        if st.button("🚀 Analisis", type="primary", use_container_width=True):
            if model is None:
                st.error("❌ Model tidak tersedia")
            else:
                with st.spinner("Menganalisis..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                        f.write(audio.getbuffer())
                        temp_path = f.name
                    
                    result, probs = predict_accent(temp_path, model)
                    
                    with col2:
                        st.subheader("📊 Hasil")
                        
                        if "Error" in result:
                            st.error(result)
                        else:
                            st.success(f"**{result}**")
                            
                            if probs:
                                st.write("**Confidence:**")
                                for name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                                    st.progress(prob/100, text=f"{name}: {prob:.1f}%")
                        
                        st.divider()
                        st.subheader("📋 Info")
                        
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
                                    st.metric("Label", info['label_aksen'])
                            else:
                                st.info("Data tidak ditemukan")
                    
                    os.unlink(temp_path)

with col2:
    if not audio:
        st.info("👆 Upload audio untuk memulai")

st.divider()
st.caption("🎯 Deteksi Aksen Indonesia")
