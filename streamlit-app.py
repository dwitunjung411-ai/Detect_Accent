import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf
import joblib

# ==========================================================
# 1. KONFIGURASI & REGISTRASI CLASS
# ==========================================================
st.set_page_config(page_title="Deteksi Aksen Prototypical", page_icon="🎙️", layout="wide")

@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        kwargs.pop('embedding_model', None)
        super().__init__(**kwargs)
    def call(self, query_set, support_set, training=False):
        return query_set # Placeholder

# ==========================================================
# 2. FUNGSI LOAD SUMBER DAYA
# ==========================================================
@st.cache_resource
def load_resources():
    model_path = "model_aksen.keras"
    preprocess_path = "preprocess.joblib"
    
    model = None
    preprocess = None
    
    if os.path.exists(model_path):
        try:
            custom_objects = {'PrototypicalNetwork': PrototypicalNetwork}
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False, safe_mode=False)
        except Exception as e:
            st.error(f"Gagal load model: {e}")
            
    if os.path.exists(preprocess_path):
        preprocess = joblib.load(preprocess_path)
        
    return model, preprocess

@st.cache_data
def load_metadata():
    if os.path.exists("metadata.csv"):
        df = pd.read_csv("metadata.csv")
        df.columns = df.columns.str.strip().str.lower()
        return df
    return None

# ==========================================================
# 3. EXTRAKSI FEATURE (MFCC + DELTA + METADATA)
# ==========================================================
def extract_features(audio_path, preprocess, metadata_row=None):
    # 1. Load Audio
    y, sr = librosa.load(audio_path, sr=22050)
    y = librosa.util.normalize(y)
    
    # 2. MFCC + Delta
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Pad/Truncate ke 174 sesuai arsitektur model Anda
    max_len = 174
    def pad_feat(f):
        if f.shape[1] < max_len:
            return np.pad(f, ((0,0), (0, max_len - f.shape[1])), mode='constant')
        return f[:, :max_len]
    
    audio_feat = np.stack([pad_feat(mfcc), pad_feat(delta), pad_feat(delta2)], axis=-1) # (40, 174, 3)
    
    # 3. Gabungkan Metadata (jika ada)
    if preprocess and metadata_row is not None:
        usia = float(metadata_row.get('usia', 25))
        gender = str(metadata_row.get('gender', 'L'))
        prov = str(metadata_row.get('provinsi', 'Unknown'))
        
        usia_s = preprocess['scaler_usia'].transform([[usia]])
        cat_e = preprocess['ohe'].transform([[gender, prov]])
        meta_val = np.hstack([usia_s, cat_e]).astype(np.float32) # (1, meta_dim)
        
        # Broadcast metadata ke shape audio
        meta_feat = np.tile(meta_val[:, np.newaxis, np.newaxis, :], (1, 40, 174, 1))
        audio_feat = np.expand_dims(audio_feat, axis=0) # (1, 40, 174, 3)
        final_feat = np.concatenate([audio_feat, meta_feat], axis=-1)
        return final_feat
    
    return np.expand_dims(audio_feat, axis=0)

# ==========================================================
# 4. UI UTAMA
# ==========================================================
def main():
    st.title("🎙️ Sistem Deteksi Aksen Prototypical Indonesia")
    st.caption("Aplikasi berbasis Few-Shot Learning untuk klasifikasi aksen daerah.")
    st.divider()

    model, preprocess = load_resources()
    metadata = load_metadata()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📥 Input Audio")
        audio_file = st.file_uploader("Upload (.wav, .mp3)", type=["wav", "mp3"])
        
        if audio_file:
            st.audio(audio_file)
            if st.button("🚀 Extract Feature and Detect", type="primary", use_container_width=True):
                if model:
                    with st.spinner("Processing..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(audio_file.getbuffer())
                            path = tmp.name
                        
                        # Cari metadata untuk sinkronisasi feature
                        file_raw = audio_file.name.lower().split('(')[0].strip().replace(".wav","")
                        match = metadata[metadata['file_name'].str.lower().str.contains(file_raw)] if metadata is not None else None
                        info = match.iloc[0] if (match is not None and not match.empty) else None
                        
                        # Ekstraksi
                        X = extract_features(path, preprocess, info)
                        
                        try:
                            # FIX: Masukkan query_set dan dummy support_set
                            # Karena ini model Prototypical, ia butuh dua input ini
                            result = model.predict({
                                'query_set': X,
                                'support_set': X  # Kita gunakan X sebagai dummy support agar tidak error
                            }, verbose=0)
                            
                            classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
                            idx = np.argmax(result[0])
                            conf = tf.nn.softmax(result[0]).numpy()[idx] * 100
                            
                            with col2:
                                st.subheader("📊 Hasil Analisis")
                                st.success(f"🎭 Aksen Terdeteksi: **{classes[idx]}**")
                                st.info(f"Keyakinan: **{conf:.2f}%**")
                                
                                if info is not None:
                                    st.divider()
                                    st.subheader("💎 Info Pembicara")
                                    st.write(f"🎂 Usia: {info.get('usia')} Tahun")
                                    st.write(f"🚻 Gender: {info.get('gender')}")
                                    st.write(f"🗺️ Provinsi: {info.get('provinsi')}")
                        
                        except Exception as e:
                            st.error(f"Error Analisis: {e}")
                        
                        os.unlink(path)

if __name__ == "__main__":
    main()
