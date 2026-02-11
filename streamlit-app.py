import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import os
import tempfile
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import keras

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

# --- 2. PERBAIKAN DATA LOADING (SOLUSI LINE 69) ---
@st.cache_resource
def load_and_preprocess_data():
    # Pastikan file ini ada di folder yang sama dengan script
    csv_path = 'metadata.csv'
    if not os.path.exists(csv_path):
        st.error("File metadata.csv tidak ditemukan!")
        st.stop()
        
    metadata = pd.read_csv(csv_path)

    def extract_mfcc_local(file_path, sr=22050, n_mfcc=40, max_len=174):
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
        except: return None

    X_audio_features, X_meta_raw, y_text = [], [], []
    
    # Proses ekstraksi (asumsi file audio ada di direktori yang sama atau path sesuai metadata)
    for i, row in metadata.iterrows():
        f_path = str(row['file_name'])
        if os.path.exists(f_path):
            feat = extract_mfcc_local(f_path)
            if feat is not None:
                X_audio_features.append(feat)
                X_meta_raw.append([row['usia'], row['gender'], row['provinsi']])
                y_text.append(row['label_aksen'])

    X_audio_features = np.array(X_audio_features, dtype=np.float32)
    X_meta_raw = np.array(X_meta_raw, dtype=object)
    y_text = np.array(y_text, dtype=str)

    # Inisialisasi Encoders
    le_y = LabelEncoder()
    y_aksen = le_y.fit_transform(y_text)
    le_gender = LabelEncoder().fit(X_meta_raw[:, 1].astype(str))
    le_provinsi = LabelEncoder().fit(X_meta_raw[:, 2].astype(str))
    scaler_usia = StandardScaler().fit(X_meta_raw[:, 0].astype(float).reshape(-1, 1))
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X_meta_raw[:, 1:3])

    # Gabungkan fitur audio + metadata (X_final)
    u_scaled = scaler_usia.transform(X_meta_raw[:, 0].reshape(-1, 1))
    c_encoded = ohe.transform(X_meta_raw[:, 1:3])
    X_meta = np.hstack([u_scaled, c_encoded]).astype(np.float32)
    
    m_b = np.repeat(X_meta[:, np.newaxis, np.newaxis, :], X_audio_features.shape[1], axis=1)
    m_b = np.repeat(m_b, X_audio_features.shape[2], axis=2)
    X_final = np.concatenate([X_audio_features, m_b], axis=-1)

    X_train, _, y_train, _ = train_test_split(X_final, y_aksen, test_size=0.2, stratify=y_aksen)
    return le_y, scaler_usia, le_gender, le_provinsi, ohe, X_train, y_train, extract_mfcc_local

# --- 3. LOAD RESOURCES & PROTOTYPES ---
le_y, scaler_usia, le_gender, le_provinsi, ohe, X_train, y_train, extract_mfcc_func = load_and_preprocess_data()

@st.cache_resource
def load_trained_model():
    # Model name updated to match instruction
    m_path = "model_detect_aksen.keras"
    return tf.keras.models.load_model(m_path, custom_objects={"PrototypicalNetwork": PrototypicalNetwork}, compile=False)

pn_model = load_trained_model()

@st.cache_resource
def compute_prototypes(_model, _X_train, _y_train, n_classes):
    # Mengambil embedding dari model (mengatasi TrackedDict)
    emb_model = _model.layers[0] if hasattr(_model, 'layers') else _model.embedding
    embeddings = emb_model(_X_train)
    prototypes = []
    for i in range(n_classes):
        mask = (_y_train == i)
        prototypes.append(tf.reduce_mean(embeddings[mask], axis=0))
    return tf.stack(prototypes)

class_prototypes = compute_prototypes(pn_model, X_train, y_train, len(le_y.classes_))

# --- 4. UI STREAMLIT ---
st.title("🎙️ Voice Accent Classification")
# (Gunakan UI sidebar dan uploader Anda di sini untuk melakukan prediksi)
