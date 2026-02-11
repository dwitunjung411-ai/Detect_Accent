import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import keras

# --- 1. REGISTRASI CLASS (SANGAT PENTING: JANGAN DIUBAH) ---
@keras.saving.register_keras_serializable()
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super().__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, x, training=False):
        # Implementasi call sederhana agar tidak error saat di-load
        if self.embedding is not None:
            return self.embedding(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        if self.embedding: 
            config.update({"embedding_model": keras.saving.serialize_keras_object(self.embedding)})
        return config

# --- 2. FUNGSI PREPROCESSING ---
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

# --- 3. LOAD RESOURCE ---
@st.cache_resource
def load_app_resources():
    # Load Metadata & Encoders
    df = pd.read_csv('metadata.csv').dropna(subset=['usia', 'gender', 'provinsi', 'label_aksen'])
    le_y = LabelEncoder().fit(df['label_aksen'].astype(str))
    le_g = LabelEncoder().fit(df['gender'].astype(str))
    le_p = LabelEncoder().fit(df['provinsi'].astype(str))
    scaler_u = StandardScaler().fit(df['usia'].values.reshape(-1, 1))
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[['gender', 'provinsi']])

    # Load Model (model_detect_aksen.keras)
    model = tf.keras.models.load_model(
        "model_aksen.keras", 
        custom_objects={"PrototypicalNetwork": PrototypicalNetwork}, 
        compile=False
    )
    return le_y, le_g, le_p, scaler_u, ohe, model

le_y, le_g, le_p, scaler_u, ohe, main_model = load_app_resources()

@st.cache_data
def load_prototypes():
    if os.path.exists("prototypes.npy"):
        return np.load("prototypes.npy")
    return None

class_prototypes = load_prototypes()

# --- 4. FUNGSI INFERENSI TAHAN ERROR ---
def get_embedding(model, x_input):
    """Fungsi sakti untuk mendapatkan embedding tanpa peduli struktur model"""
    # Coba gunakan predict()
    try: return model.predict(x_input, verbose=0)
    except: pass
    
    # Coba akses atribut .embedding
    if hasattr(model, 'embedding'):
        emb = model.embedding
        try: return emb.predict(x_input, verbose=0)
        except: 
            try: return emb(x_input)
            except: pass

    # Coba ambil layer pertama
    try: return model.layers[0](x_input)
    except: pass

    # Terakhir, coba panggil modelnya langsung
    return model(x_input)

# --- 5. UI STREAMLIT ---
st.title("🎙️ Accent Detection System")

with st.sidebar:
    st.header("Profil")
    u_in = st
