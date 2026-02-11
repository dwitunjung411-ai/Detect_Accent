import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import keras

# --- 1. REGISTRASI CLASS MODEL ---
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

# --- 2. FUNGSI PREPROCESSING AUDIO INPUT ---
def extract_mfcc(file_path, max_len=174):
    try:
        # Hanya memproses file yang di-upload user
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

# --- 3. LOAD RESOURCE (ENCODER, MODEL, & PROTOTYPES) ---
@st.cache_resource
def load_all_resources():
    # Load metadata hanya untuk inisialisasi Label & Encoder
    df = pd.read_csv('metadata.csv').dropna(subset=['usia', 'gender', 'provinsi', 'label_aksen'])
    
    le_y = LabelEncoder().fit(df['label_aksen'].astype(str))
    le_g = LabelEncoder().fit(df['gender'].astype(str))
    le_p = LabelEncoder().fit(df['provinsi'].astype(str))
    scaler_u = StandardScaler().fit(df['usia'].values.reshape(-1, 1))
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[['gender', 'provinsi']])

    # Load Model (Hanya butuh file .keras dan .npy di GitHub)
    m_path = "model_aksen.keras"
    model = tf.keras.models.load_model(m_path, custom_objects={"PrototypicalNetwork": PrototypicalNetwork}, compile=False)
    emb_layer = model.embedding if hasattr(model, 'embedding') else model.layers[0]

    # Load Prototypes (Sidik jari aksen yang sudah dihitung sebelumnya)
    if os.path.exists("prototypes.npy"):
        prototypes = np.load("prototypes.npy")
    else:
        prototypes = None
    
    return le_y, le_g, le_p, scaler_u, ohe, emb_layer, prototypes

le_y, le_g, le_p, scaler_u, ohe, emb_layer, class_prototypes = load_all_resources()

# --- 4. UI STREAMLIT ---
st.title("🎙️ Accent Detection System")

with st.sidebar:
    st.header("Profil Pengguna")
    u_in = st.number_input("Usia", 1, 100, 25)
    g_in = st.selectbox("Gender", le_g.classes_)
    p_in = st.selectbox("Provinsi", le_p.classes_)

up_file = st.file_uploader("Upload Audio Rekaman (WAV)", type=["wav"])

if up_file:
    st.audio(up_file)
    if st.button("Deteksi Sekarang"):
        if class_prototypes is None:
            st.error("⚠️ File 'prototypes.npy' tidak ditemukan! Harap upload file tersebut ke GitHub.")
            st.stop()
            
        with st.spinner("Menganalisis karakteristik suara..."):
            # Simpan file input user sementara
            with open("user_input.wav", "wb") as f: 
                f.write(up_file.getbuffer())
            
            u_feat = extract_mfcc("user_input.wav")
            
            if u_feat is not None:
                # 1. Proses Metadata Input
                meta_v = np.hstack([scaler_u.transform([[u_in]]), ohe.transform([[g_in, p_in]])]).astype(np.float32)
                
                # 2. Broadcasting Metadata ke Audio
                m_b = np.tile(meta_v, (u_feat.shape[0], u_feat.shape[1], 1))
                final_in = np.expand_dims(np.concatenate([u_feat, m_b], axis=-1), axis=0)
                
                # 3. Prediksi (Klasifikasi berdasarkan jarak terdekat)
                q_emb = tf.reshape(emb_layer(final_in), [-1]).numpy()
                dists = np.linalg.norm(class_prototypes - q_emb, axis=1)
                idx = np.argmin(dists)
                
                # 4. Tampilkan Hasil
                st.success(f"### Hasil Prediksi: Aksen {le_y.classes_[idx]}")
                st.bar_chart(pd.DataFrame({'Confidence': tf.nn.softmax(-dists).numpy()}, index=le_y.classes_))
            
            if os.path.exists("user_input.wav"): 
                os.remove("user_input.wav")

