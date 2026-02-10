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
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model
    def call(self, inputs):
        return self.embedding(inputs)

# ==========================================================
# 2. FUNGSI LOAD SUMBER DAYA
# ==========================================================
@st.cache_resource
def load_resources():
    model_path = "model_embedding_aksen.keras"
    preprocess_path = "preprocess.joblib"
    
    model = None
    preprocess = None
    
    if os.path.exists(model_path):
        try:
            custom_objects = {'PrototypicalNetwork': PrototypicalNetwork}
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False, safe_mode=False)
        except Exception as e:
            st.sidebar.error(f"Gagal load model: {e}")
            
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
# 3. EXTRAKSI FEATURE (AUDIO + SINKRONISASI METADATA)
# ==========================================================
def extract_final_features(audio_path, preprocess, info):
    # 1. Load Audio (Sesuaikan SR dengan saat training)
    y, sr = librosa.load(audio_path, sr=22050)
    y = librosa.util.normalize(y)
    
    # 2. Ekstraksi MFCC + Delta
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Padding/Truncate ke 174 (Dimensi wajib model Anda)
    max_len = 174
    def pad_feat(f):
        if f.shape[1] < max_len:
            return np.pad(f, ((0,0), (0, max_len - f.shape[1])), mode='constant')
        return f[:, :max_len]
    
    X_audio = np.stack([pad_feat(mfcc), pad_feat(delta), pad_feat(delta2)], axis=-1) # (40, 174, 3)

    # 3. Penggabungan Metadata (Usia, Gender, Provinsi)
    if preprocess and info is not None:
        try:
            usia = float(info.get('usia', 25))
            gender = str(info.get('gender', 'L'))
            prov = str(info.get('provinsi', 'Unknown'))
            
            # Transformasi menggunakan Scaler & Encoder dari joblib
            usia_s = preprocess['scaler_usia'].transform([[usia]])
            cat_e = preprocess['ohe'].transform([[gender, prov]])
            meta_val = np.hstack([usia_s, cat_e]).astype(np.float32) # (1, 8)
            
            # Broadcast agar dimensi (1, 40, 174, 8)
            X_meta = np.tile(meta_val[:, np.newaxis, np.newaxis, :], (1, 40, 174, 1))
            X_audio = np.expand_dims(X_audio, axis=0) # (1, 40, 174, 3)
            
            # Gabungkan menjadi (1, 40, 174, 11) -> 3 audio + 8 metadata
            return np.concatenate([X_audio, X_meta], axis=-1)
        except Exception as e:
            st.error(f"Gagal menggabung metadata: {e}")
            return np.expand_dims(X_audio, axis=0)
            
    return np.expand_dims(X_audio, axis=0)

# ==========================================================
# 4. UI UTAMA
# ==========================================================
def main():
    st.title("🎙️ Deteksi Aksen Indonesia (Sinkron Metadata)")
    st.divider()

    model, preprocess = load_resources()
    df_metadata = load_metadata()

    # Status Sidebar
    with st.sidebar:
        st.header("🛸 Status")
        if model: st.success("🤖 Model: Terhubung")
        else: st.error("🚫 Model: Terputus")
        if preprocess: st.success("⚙️ Preprocess: Siap")
        if df_metadata is not None: st.success("📁 Metadata: Siap")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📥 Input Audio")
        audio_file = st.file_uploader("Upload file (.wav, .mp3)", type=["wav", "mp3"])
        
        if audio_file:
            st.audio(audio_file)
            
            # Cari Metadata secara otomatis berdasarkan nama file
            user_info = None
            if df_metadata is not None:
                # Membersihkan nama file (menghapus kurung angka dan ekstensi)
                clean_name = audio_file.name.lower().split('(')[0].strip().replace(".wav","").replace(".mp3","")
                match = df_metadata[df_metadata['file_name'].str.lower().str.contains(clean_name)]
                if not match.empty:
                    user_info = match.iloc[0].to_dict()

            if st.button("🚀 Deteksi Aksen", type="primary", use_container_width=True):
                if model and user_info:
                    with st.spinner("Menganalisis Suara + Metadata..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(audio_file.getbuffer())
                            path = tmp.name
                        
                        # Ekstraksi Fitur Gabungan (11 Channel)
                        X = extract_final_features(path, preprocess, user_info)
                        
                        try:
                            # Prediksi Prototypical
                            # Kirim query_set dan support_set (menggunakan data yang sama sebagai dummy)
                            preds = model.predict({'query_set': X, 'support_set': X}, verbose=0)
                            
                            classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
                            idx = np.argmax(preds[0])
                            conf = tf.nn.softmax(preds[0]).numpy()[idx] * 100
                            
                            with col2:
                                st.subheader("📊 Hasil Analisis")
                                with st.container(border=True):
                                    st.write("### Aksen Terdeteksi:")
                                    st.info(f"**{classes[idx]}**")
                                    st.write(f"Keyakinan: {conf:.2f}%")
                                
                                st.divider()
                                st.subheader("💎 Info Pembicara (Metadata)")
                                st.markdown(f"🎂 **Usia:** {user_info.get('usia')} Tahun")
                                st.markdown(f"🚻 **Gender:** {user_info.get('gender')}")
                                st.markdown(f"🗺️ **Provinsi:** {user_info.get('provinsi')}")
                                
                        except Exception as e:
                            st.error(f"Kesalahan Prediksi: {e}")
                        
                        os.unlink(path)
                elif user_info is None:
                    st.error("❌ Nama file tidak ditemukan di metadata.csv. Harap sesuaikan nama file.")
                else:
                    st.error("❌ Model terputus. Pastikan file .keras tersedia.")

if __name__ == "__main__":
    main()
