import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import keras

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Accent Detection System", layout="centered")

# --- 2. REGISTRASI CLASS CUSTOM (WAJIB UNTUK LOAD MODEL) ---
@keras.saving.register_keras_serializable()
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super().__init__(**kwargs)
        self.embedding = embedding_model
    def get_config(self):
        config = super().get_config()
        if self.embedding: 
            config.update({"embedding_model": keras.saving.serialize_keras_object(self.embedding)})
        return config

# --- 3. FUNGSI PREPROCESSING AUDIO ---
def extract_mfcc(file_path, max_len=174):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        y = librosa.util.normalize(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        feat = np.stack([mfcc, delta, delta2], axis=-1)
        if feat.shape[1] < max_len:
            feat = np.pad(feat, ((0,0), (0, max_len - feat.shape[1]), (0,0)), mode='constant')
        else:
            feat = feat[:, :max_len, :]
        return feat
    except Exception as e:
        return None

# --- 4. LOAD SEMUA RESOURCE ---
@st.cache_resource
def load_app_resources():
    # Load Metadata & Fit Encoders
    df = pd.read_csv('metadata.csv').dropna(subset=['usia', 'gender', 'provinsi', 'label_aksen'])
    
    le_y = LabelEncoder().fit(df['label_aksen'].astype(str))
    le_g = LabelEncoder().fit(df['gender'].astype(str))
    le_p = LabelEncoder().fit(df['provinsi'].astype(str))
    scaler_u = StandardScaler().fit(df['usia'].values.reshape(-1, 1))
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[['gender', 'provinsi']])

    # Load Model (model_detect_aksen.keras)
    m_path = "model_aksen.keras"
    model = tf.keras.models.load_model(m_path, 
                                       custom_objects={"PrototypicalNetwork": PrototypicalNetwork}, 
                                       compile=False)
    emb_layer = model.embedding if hasattr(model, 'embedding') else model.layers[0]
    
    return le_y, le_g, le_p, scaler_u, ohe, emb_layer

le_y, le_g, le_p, scaler_u, ohe, emb_layer = load_app_resources()

# --- 5. FUNGSI LOAD PROTOTYPES (DARI GITHUB) ---
@st.cache_data
def get_class_prototypes():
    if os.path.exists("prototypes.npy"):
        return np.load("prototypes.npy")
    return None

class_prototypes = get_class_prototypes()

# --- 6. ANTARMUKA PENGGUNA (UI) ---
st.title("🎙️ Accent Detection System")

with st.sidebar:
    st.header("Profil Pengguna")
    u_in = st.number_input("Usia", 1, 100, 25)
    g_in = st.selectbox("Gender", le_g.classes_)
    p_in = st.selectbox("Asal Provinsi", le_p.classes_)

up_file = st.file_uploader("Upload Rekaman Suara (WAV)", type=["wav"])

if up_file:
    st.audio(up_file)
    if st.button("Deteksi Sekarang"):
        # Validasi ketersediaan file prototypes
        if class_prototypes is None:
            st.error("⚠️ File 'prototypes.npy' tidak ditemukan di GitHub! Harap reboot aplikasi jika baru saja push.")
            st.stop()
            
        with st.spinner("Menganalisis karakteristik suara..."):
            # Simpan file input sementara
            with open("temp_input.wav", "wb") as f: 
                f.write(up_file.getbuffer())
            
            u_feat = extract_mfcc("temp_input.wav")
            
            if u_feat is not None:
                # 1. Transform Metadata
                meta_v = np.hstack([scaler_u.transform([[u_in]]), ohe.transform([[g_in, p_in]])]).astype(np.float32)
                
                # 2. Gabungkan Metadata dengan Fitur Audio (Broadcasting)
                m_b = np.tile(meta_v, (u_feat.shape[0], u_feat.shape[1], 1))
                final_in = np.expand_dims(np.concatenate([u_feat, m_b], axis=-1), axis=0)
                
                # 3. Ekstrak Embedding (Inference)
                query_emb = emb_layer(final_in)
                query_vec = tf.reshape(query_emb, [-1]).numpy()
                
                # 4. Klasifikasi Berdasarkan Jarak Euclidean
                dists = np.linalg.norm(class_prototypes - query_vec, axis=1)
                idx = np.argmin(dists)
                confidence = tf.nn.softmax(-dists).numpy()

                # 5. Tampilkan Hasil
                st.success(f"### Hasil Prediksi: Aksen {le_y.classes_[idx]}")
                
                res_df = pd.DataFrame({'Confidence': confidence}, index=le_y.classes_)
                st.bar_chart(res_df)
            else:
                st.error("Gagal mengekstrak fitur audio. Pastikan file WAV valid.")
            
            # Hapus file sementara
            if os.path.exists("temp_input.wav"): 
                os.remove("temp_input.wav")

