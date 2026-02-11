import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import keras

# --- 1. REGISTRASI MODEL KUSTOM ---
@keras.saving.register_keras_serializable()
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super().__init__(**kwargs)
        self.embedding = embedding_model
    def call(self, x, training=False):
        emb = self.embedding
        if isinstance(emb, dict):
            emb = emb.get('embedding', list(emb.values())[0])
        return emb(x, training=training) if callable(emb) else x
    def get_config(self):
        config = super().get_config()
        if self.embedding: 
            config.update({"embedding_model": keras.saving.serialize_keras_object(self.embedding)})
        return config

# --- 2. FUNGSI PREPROCESSING AUDIO LOKAL ---
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
    except: return None

# --- 3. LOAD RESOURCE (REFERENSI METADATA.CSV) ---
@st.cache_resource
def load_all_resources():
    # Membaca metadata agar Encoder tahu kategori apa saja yang ada
    df = pd.read_csv('metadata.csv').dropna(subset=['usia', 'gender', 'provinsi', 'label_aksen'])
    le_y = LabelEncoder().fit(df['label_aksen'].astype(str))
    le_g = LabelEncoder().fit(df['gender'].astype(str))
    le_p = LabelEncoder().fit(df['provinsi'].astype(str))
    scaler_u = StandardScaler().fit(df['usia'].values.reshape(-1, 1))
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[['gender', 'provinsi']])
    
    # Load model_detect_aksen.keras
    model = tf.keras.models.load_model("model_aksen.keras", 
                                       custom_objects={"PrototypicalNetwork": PrototypicalNetwork}, 
                                       compile=False)
    return le_y, le_g, le_p, scaler_u, ohe, model

le_y, le_g, le_p, scaler_u, ohe, main_model = load_all_resources()

# Load Prototypes yang sudah dihitung di Colab
class_prototypes = np.load("prototypes.npy") if os.path.exists("prototypes.npy") else None

# --- 4. UI & LOGIKA DETEKSI ---
st.title("🎙️ Accent Detection System")

with st.sidebar:
    st.header("Informasi Metadata")
    # Pilihan di sini diambil otomatis dari kategori di metadata.csv
    u_val = st.number_input("Usia Anda", 1, 100, 25)
    g_val = st.selectbox("Gender", le_g.classes_)
    p_val = st.selectbox("Asal Provinsi", le_p.classes_)

# Input audio dari laptop (Lokal)
uploaded_audio = st.file_uploader("Upload Audio Rekaman dari Laptop (.WAV)", type=["wav"])

if uploaded_audio and st.button("Deteksi Sekarang"):
    if class_prototypes is None:
        st.error("File 'prototypes.npy' tidak ditemukan! Pastikan sudah di-push ke GitHub.")
    else:
        with st.spinner("Mengekstrak fitur dan mencocokkan aksen..."):
            # Simpan sementara di server untuk diproses librosa
            with open("temp_audio.wav", "wb") as f:
                f.write(uploaded_audio.getbuffer())
            
            feat = extract_mfcc("temp_audio.wav")
            if feat is not None:
                # Proses Metadata User
                m_v = np.hstack([scaler_u.transform([[u_val]]), ohe.transform([[g_val, p_val]])]).astype(np.float32)
                m_b = np.tile(m_v, (feat.shape[0], feat.shape[1], 1))
                final_in = np.expand_dims(np.concatenate([feat, m_b], axis=-1), axis=0)
                
                # Inferensi Model
                try:
                    res = main_model.predict(final_in, verbose=0)
                    if len(res.shape) > 2: res = np.mean(res, axis=(1, 2))
                    query_vec = res.flatten()
                    
                    # Hitung jarak ke tiap kategori aksen dari metadata
                    dists = np.linalg.norm(class_prototypes - query_vec, axis=1)
                    idx = np.argmin(dists)
                    
                    st.success(f"### Hasil Prediksi: Aksen {le_y.classes_[idx]}")
                    
                    # Tampilkan Grafik Confidence
                    conf = tf.nn.softmax(-dists).numpy()
                    st.bar_chart(pd.DataFrame({'Confidence': conf}, index=le_y.classes_))
                except Exception as e:
                    st.error(f"Error Klasifikasi: {e}")
            
            if os.path.exists("temp_audio.wav"):
                os.remove("temp_audio.wav")
