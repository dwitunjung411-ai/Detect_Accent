import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import keras

# --- 1. REGISTRASI CLASS CUSTOM ---
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
    df = pd.read_csv('metadata.csv').dropna(subset=['usia', 'gender', 'provinsi', 'label_aksen'])
    le_y = LabelEncoder().fit(df['label_aksen'].astype(str))
    le_g = LabelEncoder().fit(df['gender'].astype(str))
    le_p = LabelEncoder().fit(df['provinsi'].astype(str))
    scaler_u = StandardScaler().fit(df['usia'].values.reshape(-1, 1))
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[['gender', 'provinsi']])

    m_path = "model_aksen.keras"
    # Gunakan nama file model_detect_aksen.keras sesuai instruksi Anda
    model = tf.keras.models.load_model(m_path, 
                                       custom_objects={"PrototypicalNetwork": PrototypicalNetwork}, 
                                       compile=False)
    return le_y, le_g, le_p, scaler_u, ohe, model

le_y, le_g, le_p, scaler_u, ohe, main_model = load_app_resources()

@st.cache_data
def get_class_prototypes():
    if os.path.exists("prototypes.npy"):
        return np.load("prototypes.npy")
    return None

class_prototypes = get_class_prototypes()

# --- 4. UI ---
st.title("🎙️ Accent Detection System")

with st.sidebar:
    st.header("Profil Pengguna")
    u_in = st.number_input("Usia", 1, 100, 25)
    g_in = st.selectbox("Gender", le_g.classes_)
    p_in = st.selectbox("Provinsi", le_p.classes_)

up_file = st.file_uploader("Upload Rekaman Suara (WAV)", type=["wav"])

if up_file:
    st.audio(up_file)
    if st.button("Deteksi Sekarang"):
        if class_prototypes is None:
            st.error("⚠️ File 'prototypes.npy' tidak ditemukan di GitHub!")
            st.stop()
            
        with st.spinner("Menganalisis..."):
            with open("temp_input.wav", "wb") as f: f.write(up_file.getbuffer())
            u_feat = extract_mfcc("temp_input.wav")
            
            if u_feat is not None:
                m_v = np.hstack([scaler_u.transform([[u_in]]), ohe.transform([[g_in, p_in]])]).astype(np.float32)
                m_b = np.tile(m_v, (u_feat.shape[0], u_feat.shape[1], 1))
                final_in = np.expand_dims(np.concatenate([u_feat, m_b], axis=-1), axis=0).astype(np.float32)
                
                # --- PERBAIKAN: Akses layer embedding dengan aman ---
                try:
                    # Mencoba mengambil embedding model dari atribut atau layer pertama
                    if hasattr(main_model, 'embedding'):
                        # Jika main_model.embedding adalah TrackedDict, coba akses layer di dalamnya
                        emb_layer = main_model.embedding
                        if not hasattr(emb_layer, 'predict'):
                            # Fallback: gunakan model utama jika atribut embedding bermasalah
                            emb_layer = main_model.layers[0] if len(main_model.layers) > 0 else main_model
                    else:
                        emb_layer = main_model.layers[0] if len(main_model.layers) > 0 else main_model

                    # Gunakan fungsi call atau predict tergantung tipe objek
                    if hasattr(emb_layer, 'predict'):
                        query_emb = emb_layer.predict(final_in)
                    else:
                        query_emb = emb_layer(tf.constant(final_in))
                    
                    query_vec = np.array(query_emb).flatten()
                    
                    # Klasifikasi Jarak
                    dists = np.linalg.norm(class_prototypes - query_vec, axis=1)
                    idx = np.argmin(dists)
                    
                    st.success(f"### Hasil Prediksi: Aksen {le_y.classes_[idx]}")
                    
                    conf = tf.nn.softmax(-dists).numpy()
                    st.bar_chart(pd.DataFrame({'Confidence': conf}, index=le_y.classes_))
                except Exception as e:
                    st.error(f"Error inferensi: {str(e)}")
            
            if os.path.exists("temp_input.wav"): os.remove("temp_input.wav")
