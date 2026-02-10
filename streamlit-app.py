import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os

# ==========================================================
# PAKSA CLEAR CACHE - HAPUS SETELAH BERHASIL
# ==========================================================
st.cache_resource.clear()
st.cache_data.clear()

# ==========================================================
# LOAD MODEL DENGAN MULTIPLE FALLBACK
# ==========================================================
@st.cache_resource(show_spinner=False)
def load_accent_model_v2():
    import tensorflow as tf
    
    model_path = "model_embedding_aksen.keras"
    
    # Cek file ada atau tidak
    if not os.path.exists(model_path):
        st.sidebar.error(f"❌ File '{model_path}' tidak ditemukan")
        return None
    
    # Coba metode 1: Load biasa
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        st.sidebar.success("✅ Model loaded (method 1)")
        return model
    except Exception as e1:
        st.sidebar.warning(f"⚠️ Method 1 failed: {str(e1)[:80]}")
        
        # Coba metode 2: safe_mode=False
        try:
            model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
            st.sidebar.success("✅ Model loaded (method 2 - safe_mode=False)")
            return model
        except Exception as e2:
            st.sidebar.warning(f"⚠️ Method 2 failed: {str(e2)[:80]}")
            
            # Coba metode 3: dengan custom_objects kosong
            try:
                model = tf.keras.models.load_model(
                    model_path, 
                    custom_objects={},
                    compile=False
                )
                st.sidebar.success("✅ Model loaded (method 3 - custom_objects)")
                return model
            except Exception as e3:
                st.sidebar.error(f"❌ Semua metode gagal!")
                st.sidebar.error(f"Error terakhir: {str(e3)[:150]}")
                return None

# ==========================================================
# LOAD METADATA - DIPERBAIKI
# ==========================================================
@st.cache_data
def load_metadata_df():
    try:
        if os.path.exists("metadata.csv"):
            df = pd.read_csv("metadata.csv")
            return df
        else:
            return None
    except Exception as e:
        st.sidebar.warning(f"⚠️ Metadata error: {str(e)[:100]}")
        return None

# ==========================================================
# PREDIKSI
# ==========================================================
def predict_accent(audio_path, model):
    if model is None:
        return "Model tidak tersedia"
    
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        # Input
        X = np.expand_dims(mfcc_mean, axis=0)
        
        # Predict
        pred = model.predict(X, verbose=0)
        
        # Hasil
        classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        idx = np.argmax(pred[0])
        conf = pred[0][idx] * 100
        
        return f"{classes[idx]} ({conf:.1f}%)"
        
    except Exception as e:
        return f"Error: {str(e)}"

# ==========================================================
# UI
# ==========================================================
st.set_page_config(page_title="Deteksi Aksen", page_icon="🎙️", layout="wide")

st.title("🎙️ Deteksi Aksen Indonesia")
st.divider()

# Sidebar - Clear Cache Button
with st.sidebar:
    st.header("⚙️ Pengaturan")
    if st.button("🔄 Clear Cache & Reload", use_container_width=True):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    st.divider()

# Load
model = load_accent_model_v2()
metadata = load_metadata_df()

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Input Audio")
    
    audio = st.file_uploader("Upload (.wav, .mp3)", type=["wav", "mp3"])
    
    if audio:
        st.audio(audio)
        
        if st.button("🚀 Analisis", type="primary", use_container_width=True):
            if model:
                with st.spinner("Analyzing..."):
                    # Save temp
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                        f.write(audio.getbuffer())
                        path = f.name
                    
                    # Predict
                    result = predict_accent(path, model)
                    
                    # Show hasil di col2
                    with col2:
                        st.subheader("📊 Hasil")
                        
                        # Tampilkan hasil prediksi
                        if "Error" in result:
                            st.error(result)
                        else:
                            st.success(result)
                        
                        st.divider()
                        
                        # Metadata - DIPERBAIKI
                        if metadata is not None and not metadata.empty:
                            try:
                                # Cari berdasarkan nama file
                                match = metadata[metadata['file_name'] == audio.name]
                                
                                if not match.empty:
                                    info = match.iloc[0]
                                    st.write(f"🎂 Usia: {info.get('usia', '-')} Tahun")
                                    st.write(f"🚻 Gender: {info.get('gender', '-')}")
                                    st.write(f"🗺️ Provinsi: {info.get('provinsi', '-')}")
                                else:
                                    st.info("ℹ️ Data metadata tidak ditemukan untuk file ini")
                            except Exception as e:
                                st.warning(f"⚠️ Tidak bisa load metadata: {str(e)[:100]}")
                        else:
                            st.info("ℹ️ File metadata.csv tidak tersedia")
                    
                    # Cleanup
                    try:
                        os.unlink(path)
                    except:
                        pass
            else:
                st.error("❌ Model tidak tersedia. Silakan refresh halaman atau klik tombol 'Clear Cache & Reload' di sidebar.")

with col2:
    if not audio:
        st.info("👆 Upload file audio di sebelah kiri untuk memulai")
