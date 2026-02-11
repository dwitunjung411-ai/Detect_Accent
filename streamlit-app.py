import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
import keras

# Konfigurasi Halaman
st.set_page_config(page_title="Voice Accent Classification", layout="wide")

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
        distances = tf.stack(distances)
        return -distances

    def get_config(self):
        config = super().get_config()
        config.update({"embedding_model": keras.saving.serialize_keras_object(self.embedding)})
        return config

    @classmethod
    def from_config(cls, config):
        embedding_config = config.pop("embedding_model")
        embedding_model = keras.saving.deserialize_keras_object(embedding_config)
        return cls(embedding_model, **config)

# --- 2. FUNGSI EKSTRAKSI FITUR ---
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
        st.error(f"Gagal memproses audio: {e}")
        return None

# --- 3. LOAD DATA & RESOURCE (SOLUSI INDEX ERROR) ---
@st.cache_resource
def load_resources():
    csv_path = 'metadata.csv'
    if not os.path.exists(csv_path):
        st.error("Metadata.csv tidak ditemukan di root folder!")
        st.stop()

    df = pd.read_csv(csv_path)
    # Cleaning data agar tidak IndexError
    df = df.dropna(subset=['file_name', 'usia', 'gender', 'provinsi', 'label_aksen'])

    X_audio_feats, X_meta_raw, y_text = [], [], []

    for _, row in df.iterrows():
        f_path = str(row['file_name'])
        if os.path.exists(f_path):
            feat = extract_mfcc(f_path)
            if feat is not None:
                X_audio_feats.append(feat)
                X_meta_raw.append([row['usia'], str(row['gender']), str(row['provinsi'])])
                y_text.append(row['label_aksen'])

    if len(X_meta_raw) == 0:
        st.error("Data audio tidak ditemukan. Cek path file di GitHub!")
        st.stop()

    X_audio_feats = np.array(X_audio_feats, dtype=np.float32)
    X_meta_raw = np.array(X_meta_raw, dtype=object)
    y_text = np.array(y_text, dtype=str)

    # Label Encoders
    le_y = LabelEncoder().fit(y_text)
    le_gender = LabelEncoder().fit(X_meta_raw[:, 1])
    le_provinsi = LabelEncoder().fit(X_meta_raw[:, 2])
    scaler_usia = StandardScaler().fit(X_meta_raw[:, 0].reshape(-1, 1))
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X_meta_raw[:, 1:3])

    # Final X untuk training/prototypes
    u_scaled = scaler_usia.transform(X_meta_raw[:, 0].reshape(-1, 1))
    c_encoded = ohe.transform(X_meta_raw[:, 1:3])
    X_meta = np.hstack([u_scaled, c_encoded]).astype(np.float32)
    
    m_b = np.repeat(X_meta[:, np.newaxis, np.newaxis, :], X_audio_feats.shape[1], axis=1)
    m_b = np.repeat(m_b, X_audio_feats.shape[2], axis=2)
    X_final = np.concatenate([X_audio_feats, m_b], axis=-1)

    y_indices = le_y.transform(y_text)
    X_train, _, y_train, _ = train_test_split(X_final, y_indices, test_size=0.2, stratify=y_indices)

    return le_y, le_gender, le_provinsi, scaler_usia, ohe, X_train, y_train

# Load semua resource
le_y, le_gender, le_provinsi, scaler_usia, ohe, X_train, y_train = load_resources()

@st.cache_resource
def load_model():
    m_path = "model_detect_aksen.keras"
    if not os.path.exists(m_path):
        st.error("File model_detect_aksen.keras tidak ditemukan!")
        st.stop()
    return tf.keras.models.load_model(m_path, custom_objects={"PrototypicalNetwork": PrototypicalNetwork}, compile=False)

pn_model = load_model()

@st.cache_resource
def get_prototypes(_model, _X_train, _y_train, n_classes):
    emb_model = _model.layers[0] if hasattr(_model, 'layers') else _model.embedding
    embeddings = emb_model(_X_train)
    prototypes = []
    for i in range(n_classes):
        mask = (_y_train == i)
        prototypes.append(tf.reduce_mean(embeddings[mask], axis=0))
    return tf.stack(prototypes)

class_prototypes = get_prototypes(pn_model, X_train, y_train, len(le_y.classes_))

# --- 4. UI STREAMLIT ---
st.title("🎙️ Voice Accent Classification")
st.markdown("Aplikasi ini menggunakan **Prototypical Networks** untuk mendeteksi aksen berdasarkan audio dan data demografis.")

st.sidebar.header("Input Data Pendukung")
input_usia = st.sidebar.number_input("Usia", min_value=1, max_value=100, value=20)
input_gender = st.sidebar.selectbox("Gender", le_gender.classes_)
input_provinsi = st.sidebar.selectbox("Provinsi", le_provinsi.classes_)

uploaded_file = st.file_uploader("Upload File Audio (WAV)", type=["wav"])

if uploaded_file is not None:
    # Simpan file sementara
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Klasifikasi Aksen"):
        with st.spinner('Menganalisis...'):
            # 1. Ekstrak Fitur Audio
            audio_feat = extract_mfcc("temp_audio.wav")
            
            if audio_feat is not None:
                # 2. Proses Fitur Metadata
                u_s = scaler_usia.transform([[input_usia]])
                c_e = ohe.transform([[input_gender, input_provinsi]])
                meta_feat = np.hstack([u_s, c_e]).astype(np.float32)
                
                # 3. Gabungkan (Match Dimension)
                m_b = np.repeat(meta_feat[np.newaxis, np.newaxis, :], audio_feat.shape[0], axis=0)
                m_b = np.repeat(m_b, audio_feat.shape[1], axis=1)
                final_input = np.concatenate([audio_feat, m_b], axis=-1)
                final_input = np.expand_dims(final_input, axis=0) # Add batch dim

                # 4. Prediksi Jarak ke Prototypes
                emb_model = pn_model.layers[0] if hasattr(pn_model, 'layers') else pn_model.embedding
                query_embedding = emb_model(final_input)
                
                # Hitung Euclidean Distance
                dists = tf.norm(class_prototypes - query_embedding, axis=1)
                probs = tf.nn.softmax(-dists).numpy()
                pred_idx = np.argmin(dists)
                
                # 5. Tampilkan Hasil
                st.success(f"### Hasil Prediksi: Aksen {le_y.classes_[pred_idx]}")
                
                # Tampilkan Probabilitas
                chart_data = pd.DataFrame({
                    'Aksen': le_y.classes_,
                    'Confidence': probs
                })
                st.bar_chart(chart_data.set_index('Aksen'))
            
            # Hapus file temp
            if os.path.exists("temp_audio.wav"):
                os.remove("temp_audio.wav")

---
### Apa yang baru di kode ini?
1.  **Input Sidebar**: Saya menambahkan `selectbox` dan `number_input` yang datanya diambil langsung dari `LabelEncoder`. Jadi, pilihan gender dan provinsi akan otomatis mengikuti isi `metadata.csv` Anda.
2.  **Logic Prediksi**: Kode ini menghitung jarak antara *embedding* suara yang di-upload dengan *prototypes* yang sudah dihitung dari data training.
3.  **Visualisasi**: Hasil klasifikasi ditampilkan dengan Bar Chart untuk melihat tingkat kepercayaan (confidence) model terhadap tiap aksen.
4.  **Kerapihan**: File audio sementara otomatis dihapus setelah diproses untuk menjaga efisiensi storage di Streamlit Cloud.

**Langkah selanjutnya:**
Coba push kode ini ke GitHub Anda. Pastikan file `metadata.csv` dan `model_detect_aksen.keras` berada di folder yang sama dengan file `.py` ini. Apakah ada bagian dari visualisasi hasil yang ingin Anda ubah?
