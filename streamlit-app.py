import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import keras

# --- 1. DEFINISI CLASS (HARUS ADA AGAR LOAD_MODEL BERHASIL) ---
@keras.saving.register_keras_serializable()
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super().__init__(**kwargs)
        self.embedding = embedding_model
    def get_config(self):
        config = super().get_config()
        if self.embedding: config.update({"embedding_model": keras.saving.serialize_keras_object(self.embedding)})
        return config

# --- 2. LOAD RESOURCE (ENCODER & MODEL) ---
@st.cache_resource
def load_all():
    # Load metadata hanya untuk mendapatkan list label aksen
    df = pd.read_csv('metadata.csv').dropna(subset=['label_aksen', 'usia', 'gender', 'provinsi'])
    
    le_y = LabelEncoder().fit(df['label_aksen'].astype(str))
    le_g = LabelEncoder().fit(df['gender'].astype(str))
    le_p = LabelEncoder().fit(df['provinsi'].astype(str))
    scaler = StandardScaler().fit(df['usia'].values.reshape(-1, 1))
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[['gender', 'provinsi']])

    # Load Model (Gunakan nama file sesuai instruksi Anda)
    model = tf.keras.models.load_model(
        "model_aksen.keras", 
        custom_objects={"PrototypicalNetwork": PrototypicalNetwork}, 
        compile=False
    )
    return le_y, le_g, le_p, scaler, ohe, model

le_y, le_g, le_p, scaler, ohe, pn_model = load_all()

# --- 3. UI ---
st.title("🎙️ Sistem Deteksi Aksen")

with st.sidebar:
    st.header("Data Profil")
    u = st.number_input("Usia", 1, 100, 25)
    g = st.selectbox("Gender", le_g.classes_)
    p = st.selectbox("Provinsi", le_p.classes_)

up = st.file_uploader("Upload Audio (WAV)", type=["wav"])

if up:
    st.audio(up)
    if st.button("Deteksi Sekarang"):
        with st.spinner("Sedang mengklasifikasi..."):
            # Simpan dan Ekstrak MFCC
            with open("temp.wav", "wb") as f: f.write(up.getbuffer())
            
            y, sr = librosa.load("temp.wav", sr=22050)
            mfcc = librosa.feature.mfcc(y=librosa.util.normalize(y), sr=sr, n_mfcc=40)
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            
            # Padding ke 174 kolom
            feat = np.stack([mfcc, delta, delta2], axis=-1)
            if feat.shape[1] < 174:
                feat = np.pad(feat, ((0,0), (0, 174 - feat.shape[1]), (0,0)), mode='constant')
            else:
                feat = feat[:, :174, :]

            # Gabungkan dengan Metadata
            meta_v = np.hstack([scaler.transform([[u]]), ohe.transform([[g, p]])]).astype(np.float32)
            meta_b = np.tile(meta_v, (feat.shape[0], feat.shape[1], 1))
            final_in = np.expand_dims(np.concatenate([feat, meta_b], axis=-1), axis=0)

            # PREDIKSI LANGSUNG (Tanpa membandingkan dengan prototypes.npy)
            # Pastikan model Anda memang mengembalikan probabilitas kelas
            prediction = pn_model.predict(final_in)
            
            # Jika output model adalah jarak (negatif), kita ambil argmax
            idx = np.argmax(prediction) 
            
            st.success(f"### Hasil Klasifikasi: {le_y.classes_[idx]}")
            
            if os.path.exists("temp.wav"): os.remove("temp.wav")
