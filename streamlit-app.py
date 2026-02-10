import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# 1. FIX KUSTOM OBJECT & REGISTRASI MODEL
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        kwargs.pop('embedding_model', None)
        kwargs.pop('name', None) 
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return inputs

    @classmethod
    def from_config(cls, config):
        for key in ["embedding_model", "name", "trainable", "dtype"]:
            config.pop(key, None)
        return cls(**config)

# ==========================================================
# 2. FUNGSI LOAD SUMBER DAYA
# ==========================================================
@st.cache_resource
def load_accent_model():
    model_path = "model_aksen.keras"
    if not os.path.exists(model_path):
        return None
    try:
        custom_objects = {
            'PrototypicalNetwork': PrototypicalNetwork,
            'embedding_model': tf.keras.layers.Layer
        }
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects, 
            compile=False,
            safe_mode=False
        )
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

@st.cache_data
def load_metadata_df():
    if os.path.exists("metadata.csv"):
        # Load CSV dan bersihkan nama kolom dari spasi yang tidak sengaja
        df = pd.read_csv("metadata.csv")
        df.columns = df.columns.str.strip().str.lower()
        return df
    return None

# ==========================================================
# 3. FUNGSI PREDIKSI
# ==========================================================
def predict_accent(audio_path, model):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        X = np.expand_dims(mfcc_mean, axis=0)
        
        logits = model.predict(X, verbose=0)
        
        # Gunakan Softmax jika angka keyakinan terlalu tinggi/aneh
        # Ini menormalkan output menjadi rentang 0-1 (probabilitas)
        probs = tf.nn.softmax(logits).numpy()
        
        classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        idx = np.argmax(probs[0])
        conf = probs[0][idx] * 100
        
        return classes[idx], conf
    except Exception as e:
        return f"Error: {e}", 0

# ==========================================================
# 4. ANTARMUKA UTAMA (UI)
# ==========================================================
def main():
    st.set_page_config(page_title="Deteksi Aksen Indonesia", page_icon="🎙️", layout="wide")
    st.title("🎙️ Deteksi Aksen Indonesia")
    st.markdown("Unggah file audio untuk menganalisis aksen dan melihat data pembicara.")
    st.divider()

    # Load resources
    model = load_accent_model()
    metadata = load_metadata_df()

    if model:
        st.sidebar.success("✅ Model Aktif")
    else:
        st.sidebar.error("❌ Model Tidak Ditemukan")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📥 Input Audio")
        audio_file = st.file_uploader("Pilih file (.wav, .mp3)", type=["wav", "mp3"])
        
        if audio_file:
            st.audio(audio_file)
            if st.button("🚀 Jalankan Analisis", type="primary", use_container_width=True):
                with st.spinner("Menganalisis karakteristik suara..."):
                    # Simpan temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_file.getbuffer())
                        path = tmp.name
                    
                    # Prediksi
                    label, conf = predict_accent(path, model)
                    
                    with col2:
                        st.subheader("📊 Hasil Deteksi")
                        if conf > 0:
                            st.success(f"Aksen Terdeteksi: **{label}**")
                            st.info(f"Keyakinan: **{conf:.2f}%**")
                            
                            # LOGIKA METADATA
                            if metadata is not None:
                                st.divider()
                                st.subheader("👤 Profil Pembicara")
                                
                                # Pencocokan nama file yang fleksibel
                                nama_cari = audio_file.name.lower().split('(')[0].strip() # ambil nama depan sebelum tanda '('
                                
                                # Cari di kolom file_name
                                match = metadata[metadata['file_name'].astype(str).str.lower().str.contains(nama_cari)]
                                
                                if not match.empty:
                                    info = match.iloc[0]
                                    m1, m2, m3 = st.columns(3)
                                    m1.metric("Usia", f"{info.get('usia', '-')} Thn")
                                    m2.metric("Gender", info.get('gender', '-'))
                                    m3.metric("Provinsi", info.get('provinsi', '-'))
                                else:
                                    st.warning("⚠️ Metadata tidak ditemukan untuk file ini.")
                                    st.caption(f"Dicari di CSV: `{nama_cari}`")
                        else:
                            st.error(label)
                    
                    os.unlink(path)

if __name__ == "__main__":
    main()
