import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow.keras.models import Model
import keras

# Konfigurasi Halaman
st.set_page_config(page_title="Voice Accent Detection", layout="wide")

# --- 1. REGISTRASI CLASS PROTOTYPICAL NETWORK ---
@keras.saving.register_keras_serializable()
class PrototypicalNetwork(Model):
    def __init__(self, embedding_model, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, support_set, query_set, support_labels, n_way):
        support_embeddings = self.embedding(support_set)
        query_embeddings = self.embedding(query_set)
        prototypes = []
        for i in range(n_way):
            mask = tf.equal(support_labels, i)
            class_embeddings = tf.boolean_mask(support_embeddings, mask)
            prototype = tf.reduce_mean(class_embeddings, axis=0)
            prototypes.append(prototype)
        prototypes = tf.stack(prototypes)
        distances = []
        for q in query_embeddings:
            dist = tf.norm(prototypes - q, axis=1)
            distances.append(dist)
        return -tf.stack(distances)

    def get_config(self):
        config = super().get_config()
        config.update({"embedding_model": keras.saving.serialize_keras_object(self.embedding)})
        return config

# --- 2. FUNGSI EKSTRAKSI MFCC (Hanya untuk Audio Input) ---
def extract_mfcc(file_path, sr=22050, n_mfcc=40, max_len=174):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        y = librosa.util.normalize(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            delta = np.pad(delta, ((0, 0), (0, pad_width)), mode='constant')
            delta2 = np.pad(delta2, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
            delta = delta[:, :max_len]
            delta2 = delta2[:, :max_len]
        return np.stack([mfcc, delta, delta2], axis=-1)
    except Exception as e:
        st.error(f"Error ekstraksi audio: {e}")
        return None

# --- 3. LOAD ENCODERS (Hanya dari CSV, Tanpa Baca Audio GitHub) ---
@st.cache_resource
def load_encoders():
    csv_path = 'metadata.csv'
    if not os.path.exists(csv_path):
        st.error("metadata.csv tidak ditemukan!")
        st.stop()
    
    df = pd.read_csv(csv_path)
    # Fit encoders berdasarkan isi teks di CSV
    le_y = LabelEncoder().fit(df['label_aksen'].astype(str))
    le_gender = LabelEncoder().fit(df['gender'].astype(str))
    le_provinsi = LabelEncoder().fit(df['provinsi'].astype(str))
    scaler_usia = StandardScaler().fit(df['usia'].values.reshape(-1, 1))
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[['gender', 'provinsi']])
    
    return le_y, le_gender, le_provinsi, scaler_usia, ohe

le_y, le_gender, le_provinsi, scaler_usia, ohe = load_encoders()

# --- 4. LOAD MODEL & PROTOTYPES ---
@st.cache_resource
def load_model_and_prototypes():
    m_path = "model_detect_aksen.keras"
    model = tf.keras.models.load_model(m_path, custom_objects={"PrototypicalNetwork": PrototypicalNetwork}, compile=False)
    
    # Karena audio training tidak ada di GitHub, kita asumsikan model sudah membawa weight.
    # Jika Anda punya file 'prototypes.npy', load dari situ. 
    # Jika tidak, kita buat dummy prototypes atau load dari file eksternal.
    # Di sini saya asumsikan model.embedding adalah layer pertama.
    emb_model = model.embedding if hasattr(model, 'embedding') else model.layers[0]
    
    return model, emb_model

pn_model, embedding_layer = load_model_and_prototypes()

# --- 5. UI STREAMLIT ---
st.title("🎙️ Accent & Demographics Detector")
st.info("Catatan: Model akan mendeteksi aksen berdasarkan fitur suara dan data profil yang Anda masukkan.")

with st.sidebar:
    st.header("Profil Pengguna")
    in_usia = st.number_input("Masukkan Usia", 1, 100, 25)
    in_gender = st.selectbox("Pilih Gender", le_gender.classes_)
    in_provinsi = st.selectbox("Asal Provinsi", le_provinsi.classes_)

uploaded_file = st.file_uploader("Upload Rekaman Suara (WAV)", type=["wav"])

if uploaded_file:
    # Tampilkan player audio
    st.audio(uploaded_file)
    
    if st.button("Mulai Deteksi"):
        with st.spinner("Menganalisis karakteristik suara..."):
            # Simpan file sementara untuk diproses librosa
            with open("input_user.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 1. Ekstrak Audio
            audio_feat = extract_mfcc("input_user.wav")
            
            if audio_feat is not None:
                # 2. Proses Metadata Input
                meta_u = scaler_usia.transform([[in_usia]])
                meta_c = ohe.transform([[in_gender, in_provinsi]])
                meta_final = np.hstack([meta_u, meta_c]).astype(np.float32)
                
                # 3. Gabungkan Audio + Metadata (Broadcasting)
                # Menyamakan dimensi metadata dengan dimensi MFCC (40, 174)
                m_broad = np.tile(meta_final, (audio_feat.shape[0], audio_feat.shape[1], 1))
                combined_feat = np.concatenate([audio_feat, m_broad], axis=-1)
                combined_feat = np.expand_dims(combined_feat, axis=0) # Add batch dim

                # 4. Prediksi
                # Mendapatkan embedding dari suara yang di-upload
                query_embedding = embedding_layer(combined_feat)
                query_vec = tf.reshape(query_embedding, [query_embedding.shape[-1]]).numpy()

                # PENTING: Karena audio training tidak di GitHub, Anda harus menyediakan
                # file 'prototypes.npy' yang sudah disimpan saat training selesai.
                if os.path.exists("prototypes.npy"):
                    all_prototypes = np.load("prototypes.npy")
                    
                    # Hitung jarak ke tiap prototype
                    dists = np.linalg.norm(all_prototypes - query_vec, axis=1)
                    pred_idx = np.argmin(dists)
                    conf = tf.nn.softmax(-dists).numpy()

                    # 5. Tampilkan Hasil
                    st.divider()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Prediksi Aksen", le_y.classes_[pred_idx])
                        st.write(f"Confidence: {np.max(conf)*100:.2f}%")
                    
                    with col2:
                        # Bar chart probabilitas
                        res_df = pd.DataFrame({'Aksen': le_y.classes_, 'Skor': conf})
                        st.bar_chart(res_df.set_index('Aksen'))
                else:
                    st.warning("File 'prototypes.npy' tidak ditemukan. Pastikan Anda mengunggah file prototype hasil training ke GitHub.")

            # Cleanup
            if os.path.exists("input_user.wav"):
                os.remove("input_user.wav")
