import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import keras

# --- 1. REGISTRASI CLASS CUSTOM (SOLUSI: method not implemented) ---
@keras.saving.register_keras_serializable()
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super().__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, x, training=False):
        # Solusi untuk TrackedDict: mencari layer asli di dalam dictionary
        emb_layer = self.embedding
        if isinstance(emb_layer, dict):
            emb_layer = emb_layer.get('embedding', list(emb_layer.values())[0])
        
        if callable(emb_layer):
            return emb_layer(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        if self.embedding: 
            config.update({"embedding_model": keras.saving.serialize_keras_object(self.embedding)})
        return config

# --- 2. FUNGSI PREPROCESSING AUDIO ---
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

# --- 3. LOAD RESOURCE (ENCODER & MODEL) ---
@st.cache_resource
def load_app_resources():
    # Membaca metadata agar Encoder tahu kategori apa saja yang ada
    df = pd.read_csv('metadata.csv').dropna(subset=['usia', 'gender', 'provinsi', 'label_aksen'])
    le_y = LabelEncoder().fit(df['label_aksen'].astype(str))
    le_g = LabelEncoder().fit(df['gender'].astype(str))
    le_p = LabelEncoder().fit(df['provinsi'].astype(str))
    scaler_u = StandardScaler().fit(df['usia'].values.reshape(-1, 1))
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[['gender', 'provinsi']])

    # Load model_detect_aksen.keras sesuai koreksi user
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

# --- 4. FUNGSI INFERENSI (SOLUSI FINAL: TrackedDict & Shape Mismatch) ---
def get_embedding_safely(model, x_input):
    """Memastikan output model diringkas menjadi vektor fitur tunggal (128,)"""
    x_tensor = tf.convert_to_tensor(x_input, dtype=tf.float32)
    try:
        # Coba cara standar predict()
        res = model.predict(x_tensor, verbose=0)
    except:
        # Fallback jika model adalah TrackedDict
        res = model(x_tensor, training=False).numpy()
    
    # Global Average Pooling manual jika output masih berupa peta fitur (spasial)
    # Ini yang mencegah error (11,) atau (76560,)
    if len(res.shape) > 2:
        res = np.mean(res, axis=(1, 2))
        
    return np.reshape(res, (1, -1))

# --- 5. ANTARMUKA PENGGUNA (UI) ---
st.title("🎙️ Accent Detection System")

with st.sidebar:
    st.header("Informasi Pengguna")
    u_in = st.number_input("Usia", 1, 100, 25)
    g_in = st.selectbox("Gender", le_g.classes_)
    p_in = st.selectbox("Asal Provinsi", le_p.classes_)

up_file = st.file_uploader("Upload Audio Rekaman (WAV)", type=["wav"])

if up_file:
    st.audio(up_file)
    if st.button("Deteksi Sekarang"):
        # Validasi file prototypes.npy di GitHub
        if class_prototypes is None:
            st.error("⚠️ File 'prototypes.npy' tidak ditemukan! Harap upload file tersebut ke GitHub.")
            st.stop()
            
        with st.spinner("Menganalisis karakteristik suara..."):
            # Simpan sementara audio dari lokal
            with open("temp.wav", "wb") as f: 
                f.write(up_file.getbuffer())
            
            u_feat = extract_mfcc("temp.wav")
            
            if u_feat is not None:
                # Meta Processing (Broadcasting 11 fitur metadata)
                m_v = np.hstack([scaler_u.transform([[u_in]]), ohe.transform([[g_in, p_in]])]).astype(np.float32)
                m_b = np.tile(m_v, (u_feat.shape[0], u_feat.shape[1], 1))
                
                # Gabungkan Audio + Metadata sebelum masuk model
                final_in = np.expand_dims(np.concatenate([u_feat, m_b], axis=-1), axis=0)
                
                try:
                    # Ambil Embedding 128-dimensi
                    query_vec = get_embedding_safely(main_model, final_in)
                    
                    # HITUNG JARAK: Bandingkan (5, 128) dengan vektor user (128,)
                    dists = np.linalg.norm(class_prototypes - query_vec.flatten(), axis=1)
                    idx = np.argmin(dists)
                    
                    st.success(f"### Hasil Prediksi: Aksen {le_y.classes_[idx]}")
                    
                    # Chart Confidence
                    conf = tf.nn.softmax(-dists).numpy()
                    st.bar_chart(pd.DataFrame({'Confidence': conf}, index=le_y.classes_))
                except Exception as e:
                    st.error(f"Gagal melakukan klasifikasi: {str(e)}")
            
            if os.path.exists("temp.wav"): 
                os.remove("temp.wav")
