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
    
    model_path = "model_aksen.keras"
    
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
# LOAD METADATA - DIPERBAIKI TOTAL
# ==========================================================
@st.cache_data
def load_metadata_df():
    try:
        metadata_path = "metadata.csv"
        if os.path.exists(metadata_path):
            # Load CSV dengan encoding yang benar
            df = pd.read_csv(metadata_path, encoding='utf-8')
            
            # Hapus baris kosong jika ada
            df = df.dropna(subset=['file_name'])
            
            # Strip whitespace dari kolom file_name
            df['file_name'] = df['file_name'].str.strip()
            
            return df
        else:
            return None
    except Exception as e:
        st.sidebar.warning(f"⚠️ Metadata load error: {str(e)[:150]}")
        return None

# ==========================================================
# PREDIKSI
# ==========================================================
def predict_accent(audio_path, model):
    if model is None:
        return "Model tidak tersedia", None
    
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
        
        # Return hasil prediksi dan array probabilitas
        result_text = f"{classes[idx]} ({conf:.1f}%)"
        
        # Buat dictionary untuk semua kelas dengan confidence
        all_probs = {classes[i]: float(pred[0][i] * 100) for i in range(len(classes))}
        
        return result_text, all_probs
        
    except Exception as e:
        return f"Error: {str(e)}", None

# ==========================================================
# FUNGSI CARI METADATA
# ==========================================================
def get_metadata_info(filename, metadata_df):
    """Cari metadata berdasarkan nama file"""
    if metadata_df is None or metadata_df.empty:
        return None
    
    try:
        # Cari exact match
        match = metadata_df[metadata_df['file_name'] == filename]
        
        if not match.empty:
            return match.iloc[0].to_dict()
        else:
            return None
    except Exception as e:
        st.warning(f"Error saat mencari metadata: {str(e)[:100]}")
        return None

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

# Debug info di sidebar
if metadata is not None:
    st.sidebar.info(f"📊 Metadata: {len(metadata)} records loaded")

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
                    result, all_probs = predict_accent(path, model)
                    
                    # Show hasil di col2
                    with col2:
                        st.subheader("📊 Hasil Prediksi")
                        
                        # Tampilkan hasil prediksi utama
                        if "Error" in result:
                            st.error(result)
                        else:
                            st.success(f"**{result}**")
                        
                        # Tampilkan detail probabilitas semua kelas
                        if all_probs:
                            st.write("**Detail Confidence:**")
                            for class_name, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                                st.progress(prob / 100.0, text=f"{class_name}: {prob:.1f}%")
                        
                        st.divider()
                        
                        # Metadata
                        st.subheader("📋 Info Speaker")
                        
                        metadata_info = get_metadata_info(audio.name, metadata)
                        
                        if metadata_info:
                            col_info1, col_info2 = st.columns(2)
                            
                            with col_info1:
                                st.metric("Usia", f"{metadata_info.get('usia', '-')} Tahun")
                                st.metric("Gender", metadata_info.get('gender', '-'))
                            
                            with col_info2:
                                st.metric("Provinsi", metadata_info.get('provinsi', '-'))
                                st.metric("Label Aksen", metadata_info.get('label_aksen', '-'))
                        else:
                            st.info("ℹ️ Data metadata tidak ditemukan untuk file ini")
                    
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

# Footer
st.divider()
st.caption("🎯 Sistem Deteksi Aksen Bahasa Indonesia | Powered by Deep Learning")
