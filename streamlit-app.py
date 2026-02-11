import streamlit as st
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import tensorflow as tf
import os
import tempfile
import random
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
        # Implementasi sesuai arsitektur di notebook pelatihan
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

        logits = -distances
        return logits

    def get_config(self):
        config = super(PrototypicalNetwork, self).get_config()
        config.update({
            "embedding_model": keras.saving.serialize_keras_object(self.embedding)
        })
        return config

    @classmethod
    def from_config(cls, config):
        embedding_config = config.pop("embedding_model")
        embedding_model = keras.saving.deserialize_keras_object(embedding_config)
        return cls(embedding_model, **config)

# --- 2. PERBAIKAN FUNGSI DATA LOADING (LINE 69) ---
@st.cache_resource
def load_and_preprocess_data():
    # Tentukan path folder dataset suara Anda
    dataset_path = "dataset_audio" 
    csv_filename = "metadata.csv"

    if not os.path.exists(csv_filename):
        st.error(f"File {csv_filename} tidak ditemukan!")
        st.stop()

    metadata = pd.read_csv(csv_filename)

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
        except Exception:
            return None

    X_audio_features, X_meta_raw, y_text = [], [], []

    # Iterasi metadata untuk ekstraksi fitur
    for i, row in metadata.iterrows():
        file_name = str(row['file_name'])
        # Cek file di direktori lokal atau path langsung
        file_path = os.path.join(dataset_path, file_name) if os.path.isdir(dataset_path) else file_name

        if os.path.exists(file_path):
            feat = extract_mfcc_local(file_path)
            if feat is not None:
                X_audio_features.append(feat)
                X_meta_raw.append([row['usia'], row['gender'], row['provinsi']])
                y_text.append(row['label_aksen'])

    X_audio_features = np.array(X_audio_features, dtype=np.float32)
    X_meta_raw = np.array(X_meta_raw, dtype=object)
    y_text = np.array(y_text, dtype=str)

    # Inisialisasi Encoders dan Scalers
    le_y = LabelEncoder()
    y_aksen = le_y.fit_transform(y_text)

    le_gender = LabelEncoder().fit(X_meta_raw[:, 1].astype(str))
    le_provinsi = LabelEncoder().fit(X_meta_raw[:, 2].astype(str))
    scaler_usia = StandardScaler().fit(X_meta_raw[:, 0].astype(float).reshape(-1, 1))
    
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(np.hstack([X_meta_raw[:, 1].reshape(-1,1), X_meta_raw[:, 2].reshape(-1,1)]))

    # Proses penggabungan fitur audio dan metadata (X_final)
    usia_scaled = scaler_usia.transform(X_meta_raw[:, 0].reshape(-1, 1))
    cat_encoded = ohe.transform(np.hstack([X_meta_raw[:, 1].reshape(-1,1), X_meta_raw[:, 2].reshape(-1,1)]))
    X_meta = np.hstack([usia_scaled, cat_encoded]).astype(np.float32)

    meta_b = np.repeat(X_meta[:, np.newaxis, np.newaxis, :], X_audio_features.shape[1], axis=1)
    meta_b = np.repeat(meta_b, X_audio_features.shape[2], axis=2)
    X_final = np.concatenate([X_audio_features, meta_b], axis=-1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_aksen, test_size=0.2, random_state=42, stratify=y_aksen
    )

    return le_y, scaler_usia, le_gender, le_provinsi, ohe, X_train, y_train, extract_mfcc_local

# --- 3. EKSEKUSI LOADING ---
# Panggil fungsi yang sudah diperbaiki
le_y, scaler_usia, le_gender, le_provinsi, ohe, X_train, y_train, extract_mfcc_func = load_and_preprocess_data()

@st.cache_resource
def load_my_model(model_path):
    # Gunakan nama file sesuai instruksi: model_detect_aksen.keras
    try:
        custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

pn_model = load_my_model("model_detect_aksen.keras")

# --- 4. PRE-COMPUTE PROTOTYPES ---
@st.cache_resource
def get_prototypes(_model, _X_train, _y_train, _n_classes):
    # Hitung rata-rata embedding untuk setiap kelas aksen
    embeddings = _model.embedding(_X_train)
    prototypes = []
    for i in range(_n_classes):
        mask = (_y_train == i)
        prototypes.append(tf.reduce_mean(embeddings[mask], axis=0))
    return tf.stack(prototypes)

if pn_model:
    class_prototypes = get_prototypes(pn_model, X_train, y_train, len(le_y.classes_))
