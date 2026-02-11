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
        # Menangani berbagai kemungkinan struktur penyimpanan model oleh Keras
        emb_layer = self.embedding
        if isinstance(emb_layer, dict):
            # Jika berupa TrackedDict, ambil layer asli di dalamnya
            emb_layer = emb_layer.get('embedding', list(emb_layer.values())[0])
        
        # Eksekusi embedding
        if callable(emb_layer):
            return emb_layer(x, training=training)
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
        
        # Padding/Truncating ke panjang standar
        if feat.shape[1] < max_len:
            feat = np.pad(feat, ((0,0), (0, max_len - feat.shape[1]), (0,0)), mode='constant')
        else:
            feat = feat[:, :max_len, :]
        return feat
    except: return None

# --- 3. LOAD RESOURCE ---
@st.cache_resource
def load_app_resources():
    # Load Metadata untuk inisialisasi Encoders
    df = pd.read_csv('metadata.csv').dropna(subset=['usia', 'gender', 'provinsi', 'label_aksen'])
    le_y = LabelEncoder().fit(df['label_aksen'].astype(str))
    le_g = LabelEncoder().fit(df['gender'].astype(str))
    le_p = LabelEncoder().fit(df['provinsi'].astype(str))
    scaler_u = StandardScaler().fit(df['usia'].values.reshape(-1, 1))
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[['gender', 'provinsi']])

    # Nama file model harus sesuai: model_detect_aksen.keras
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

# --- 4. FUNGSI INFERENSI (SOLUSI: Shape Mismatch) ---
def get_embedding_safely(model, x_input):
    x_tensor = tf.convert_to_tensor(x_input, dtype=tf.float32)
    try:
        # Gunakan predict() untuk stabilitas di server Streamlit
        res = model.predict(x_tensor, verbose=0)
        
        # FIX: Ringkas dimensi jika model mengembalikan urutan fitur temporal
        # Mengubah shape (1, 40, 174, 11) menjadi vektor tunggal (misal: 128)
        if len(res.shape) > 2:
            res = np.mean(res, axis=(1, 2))
            
        return np.reshape(res, (1, -1)) 
    except Exception as e:
        raise Exception(f"Gagal ekstraksi embedding: {str(e)}")

# --- 5. ANTARMUKA PENGGUNA (UI) ---
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
            st.error("⚠️ File 'prototypes.npy' tidak ditemukan di GitHub!")
            st.stop()
            
        with st.spinner("Menganalisis karakteristik suara..."):
            with open("temp.wav", "wb") as f: f.write(up_file.getbuffer())
            u_feat = extract_mfcc("temp.wav")
            
            if u_feat is not None:
                # Meta Processing (Broadcasting metadata ke fitur audio)
                m_v = np.hstack([scaler_u.transform([[u_in]]), ohe.transform([[g_in, p_in]])]).astype(np.float32)
                m_b = np.tile(m_v, (u_feat.shape[0], u_feat.shape[1], 1))
                final_in = np.expand_dims(np.concatenate([u_feat, m_b], axis=-1), axis=0)
                
                try:
                    # Ambil Embedding tunggal dari input user
                    query_vec = get_embedding_safely(main_model, final_in)
                    
                    # HITUNG JARAK EUCLIDEAN
                    # Bandingkan query_vec (128,) dengan class_prototypes (5, 128)
                    dists = np.linalg.norm(class_prototypes - query_vec.flatten(), axis=1)
                    idx = np.argmin(dists)
                    
                    st.success(f"### Hasil Prediksi: Aksen {le_y.classes_[idx]}")
                    
                    # Tampilkan Grafik Bar Probabilitas
                    conf = tf.nn.softmax(-dists).numpy()
                    st.bar_chart(pd.DataFrame({'Confidence': conf}, index=le_y.classes_))
                except Exception as e:
                    st.error(f"Gagal melakukan klasifikasi: {str(e)}")
            
            # Bersihkan file sementara
            if os.path.exists("temp.wav"): os.remove("temp.wav")

