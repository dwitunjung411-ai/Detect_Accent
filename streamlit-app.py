import io
import os
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import librosa
import tensorflow as tf
from pathlib import Path

# =========================================================
# CONFIG & CONSTANTS
# =========================================================
st.set_page_config(page_title="Deteksi Aksen Prototypical", page_icon="🎙️", layout="wide")

SR_DEFAULT = 22050
N_MFCC = 40
MAX_LEN = 174
MODEL_FILE = "model_aksen.keras"
PREPROCESS_FILE = "preprocess.joblib"
METADATA_FILE = "metadata.csv"

# =========================================================
# CUSTOM MODEL CLASS
# =========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        kwargs.pop('embedding_model', None)
        super().__init__(**kwargs)
    
    def call(self, inputs, training=False):
        # inputs akan berupa dictionary {'query_set': ..., 'support_set': ...}
        return inputs['query_set']

# =========================================================
# LOAD RESOURCES
# =========================================================
@st.cache_resource
def load_assets():
    model = None
    preprocess = None
    if os.path.exists(MODEL_FILE):
        try:
            custom_objects = {'PrototypicalNetwork': PrototypicalNetwork}
            model = tf.keras.models.load_model(MODEL_FILE, custom_objects=custom_objects, compile=False, safe_mode=False)
        except Exception as e:
            st.error(f"Gagal load model: {e}")
    
    if os.path.exists(PREPROCESS_FILE):
        preprocess = joblib.load(PREPROCESS_FILE)
    return model, preprocess

@st.cache_data
def load_metadata():
    if os.path.exists(METADATA_FILE):
        df = pd.read_csv(METADATA_FILE)
        df.columns = df.columns.str.strip().str.lower()
        return df
    return None

# =========================================================
# FEATURE EXTRACTION PIPELINE
# =========================================================
def get_features(audio_path, preprocess, info):
    # 1. Audio Processing
    y, sr = librosa.load(audio_path, sr=SR_DEFAULT)
    y = librosa.util.normalize(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=2048, hop_length=512)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Padding/Truncate ke MAX_LEN (174)
    def adjust_shape(feat):
        if feat.shape[1] < MAX_LEN:
            return np.pad(feat, ((0, 0), (0, MAX_LEN - feat.shape[1])), mode='constant')
        return feat[:, :MAX_LEN]

    X_audio = np.stack([adjust_shape(mfcc), adjust_shape(delta), adjust_shape(delta2)], axis=-1)
    X_audio = np.expand_dims(X_audio, axis=0) # (1, 40, 174, 3)

    # 2. Metadata Processing (Broadcast ke audio channels)
    if preprocess and info is not None:
        usia = float(info.get('usia', 25))
        gender = str(info.get('gender', 'L'))
        prov = str(info.get('provinsi', 'Unknown'))
        
        usia_s = preprocess['scaler_usia'].transform([[usia]])
        cat_e = preprocess['ohe'].transform([[gender, prov]])
        meta_combined = np.hstack([usia_s, cat_e]).astype(np.float32)
        
        # Broadcast metadata agar match dengan shape audio (1, 40, 174, 8)
        X_meta = np.tile(meta_combined[:, np.newaxis, np.newaxis, :], (1, N_MFCC, MAX_LEN, 1))
        X_final = np.concatenate([X_audio, X_meta], axis=-1)
        return X_final
    
    return X_audio

# =========================================================
# MAIN UI
# =========================================================
def main():
    st.title("🎙️ Sistem Deteksi Aksen Prototypical Indonesia")
    st.markdown("---")

    model, preprocess = load_assets()
    metadata = load_metadata()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📥 Input Audio")
        audio_file = st.file_uploader("Pilih file audio", type=["wav", "mp3"])
        
        if audio_file:
            st.audio(audio_file)
            if st.button("🚀 Jalankan Analisis", type="primary", use_container_width=True):
                if model is None:
                    st.error("Model tidak tersedia.")
                    return

                with st.spinner("Mengekstrak fitur dan menghitung jarak prototypical..."):
                    # Simpan sementara
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_file.getbuffer())
                        tmp_path = tmp.name

                    # Ambil info metadata berdasarkan nama file
                    clean_name = audio_file.name.lower().split('(')[0].strip().replace(".wav","").replace(".mp3","")
                    match = metadata[metadata['file_name'].str.lower().str.contains(clean_name)] if metadata is not None else None
                    info = match.iloc[0] if (match is not None and not match.empty) else None

                    # Ekstraksi Fitur
                    X = get_features(tmp_path, preprocess, info)

                    try:
                        # SOLUSI: Mengirimkan query_set dan support_set sebagai dictionary
                        # Kita gunakan X sebagai dummy support agar dimensi terpenuhi
                        preds = model.predict({
                            'query_set': X,
                            'support_set': X 
                        }, verbose=0)

                        # Mapping hasil
                        classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
                        idx = np.argmax(preds[0])
                        # Normalisasi skor ke persentase menggunakan Softmax
                        confidence = tf.nn.softmax(preds[0]).numpy()[idx] * 100

                        with col2:
                            st.subheader("📊 Hasil Analisis")
                            st.success(f"Aksen Terdeteksi: **{classes[idx]}**")
                            st.info(f"Keyakinan: **{confidence:.2f}%**")
                            
                            if info is not None:
                                st.divider()
                                st.subheader("👤 Profil Pembicara")
                                m1, m2, m3 = st.columns(3)
                                m1.metric("Usia", f"{info.get('usia')} Thn")
                                m2.metric("Gender", info.get('gender'))
                                m3.metric("Provinsi", info.get('provinsi'))

                    except Exception as e:
                        st.error(f"Kesalahan pada model: {e}")
                    
                    os.unlink(tmp_path)

if __name__ == "__main__":
    main()
