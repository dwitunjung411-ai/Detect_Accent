import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow.keras.models import Model
import keras

# --- 1. REGISTRASI CLASS PROTOTYPICAL NETWORK (HARUS PERSIS) ---
@keras.saving.register_keras_serializable()
class PrototypicalNetwork(Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, support_set, query_set, support_labels, n_way):
        # Logika call tetap ada agar model bisa di-load sempurna
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
        # Serialize embedding model jika ada
        if self.embedding:
            config.update({"embedding_model": keras.saving.serialize_keras_object(self.embedding)})
        return config

    @classmethod
    def from_config(cls, config):
        embedding_config = config.pop("embedding_model", None)
        embedding_model = None
        if embedding_config:
            embedding_model = keras.saving.deserialize_keras_object(embedding_config)
        return cls(embedding_model, **config)

# --- 2. FUNGSI EKSTRAKSI MFCC ---
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
        return None

# --- 3. LOAD ENCODERS ---
@st.cache_resource
def load_resources():
    csv_path = 'metadata.csv'
    if not os.path.exists(csv_path):
        st.error("metadata.csv tidak ditemukan!")
        st.stop()
    
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['usia', 'gender', 'provinsi', 'label_aksen'])
    
    le_y = LabelEncoder().fit(df['label_aksen'].astype(str))
    le_gender = LabelEncoder().fit(df['gender'].astype(str))
    le_provinsi = LabelEncoder().fit(df['provinsi'].astype(str))
    scaler_usia = StandardScaler().fit(df['usia'].values.reshape(-1, 1))
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[['gender', 'provinsi']])
    
    return le_y, le_gender, le_provinsi, scaler_usia, ohe

le_y, le_gender, le_provinsi, scaler_usia, ohe = load_resources()

# --- 4. LOAD MODEL (SOLUSI VALUE ERROR) ---
@st.cache_resource
def load_model_safely():
    m_path = "model_aksen.keras"
    if not os.path.exists(m_path):
        st.error(f"File {m_path} tidak ditemukan!")
        st.stop()
    
    try:
        # Menambahkan safe_mode=False seringkali diperlukan untuk custom model di Keras 3
        model = tf.keras.models.load_model(
            m_path, 
            custom_objects={"PrototypicalNetwork": PrototypicalNetwork}, 
            compile=False
        )
        # Ambil layer embedding (biasanya layer pertama atau atribut .embedding)
        if hasattr(model, 'embedding'):
            emb_layer = model.embedding
        else:
            emb_layer = model.layers[0]
        return model, emb_layer
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        st.stop()

pn_model, embedding_layer = load_model_safely()

# --- 5. UI STREAMLIT ---
st.title("🎙️ Accent Detection System")

with st.sidebar:
    st.header("Profil Pengguna")
    in_usia = st.number_input("Usia", 1, 100, 25)
    in_gender = st.selectbox("Gender", le_gender.classes_)
    in_provinsi = st.selectbox("Provinsi", le_provinsi.classes_)

uploaded_file = st.file_uploader("Upload Audio (WAV)", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)
    if st.button("Deteksi Sekarang"):
        with st.spinner("Memproses fitur suara..."):
            with open("temp_in.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            feat = extract_mfcc("temp_in.wav")
            if feat is not None:
                # Meta processing
                m_u = scaler_usia.transform([[in_usia]])
                m_c = ohe.transform([[in_gender, in_provinsi]])
                meta = np.hstack([m_u, m_c]).astype(np.float32)
                
                # Broadcasting
                m_b = np.tile(meta, (feat.shape[0], feat.shape[1], 1))
                final_in = np.concatenate([feat, m_b], axis=-1)
                final_in = np.expand_dims(final_in, axis=0)

                # Inference
                query_emb = embedding_layer(final_in)
                query_vec = tf.reshape(query_emb, [-1]).numpy()

                # LOGIKA PROTOTYPES
                # Jika file prototypes.npy tidak ada, Anda harus mengunggahnya!
                if os.path.exists("prototypes.npy"):
                    prototypes = np.load("prototypes.npy")
                    dists = np.linalg.norm(prototypes - query_vec, axis=1)
                    idx = np.argmin(dists)
                    probs = tf.nn.softmax(-dists).numpy()

                    st.success(f"### Hasil: Aksen {le_y.classes_[idx]}")
                    chart_data = pd.DataFrame({'Aksen': le_y.classes_, 'Skor': probs}).set_index('Aksen')
                    st.bar_chart(chart_data)
                else:
                    st.warning("Aplikasi butuh file 'prototypes.npy' di GitHub untuk membandingkan suara.")
            
            if os.path.exists("temp_in.wav"): os.remove("temp_in.wav")

