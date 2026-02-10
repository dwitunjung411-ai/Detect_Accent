import streamlit as st
import tensorflow as tf
import os
import numpy as np
import librosa

# 1. DEFINISI CUSTOM CLASS
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, support_set, query_set, support_labels, n_way):
        return self.embedding(query_set)

    def get_config(self):
        config = super().get_config()
        config.update({"embedding_model": tf.keras.layers.serialize(self.embedding)})
        return config

# 2. FUNGSI PREPROCESSING AUDIO (Agar ada Output)
def prepare_audio(uploaded_file):
    # Load audio
    y, sr = librosa.load(uploaded_file, sr=16000)
    # Ekstraksi Mel-Spectrogram (Sesuaikan dengan setting saat skripsi/training)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Reshape agar sesuai input CNN (Contoh: [1, 128, length, 1])
    # Sesuaikan dimensi ini dengan input_shape model kamu!
    features = np.expand_dims(log_mel_spec, axis=0) 
    features = np.expand_dims(features, axis=-1)
    return features

# 3. FUNGSI LOAD MODEL
@st.cache_resource
def load_accent_model():
    model_name = "model_aksen.keras"
    if not os.path.exists(model_name):
        return None
    try:
        custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
        # Memuat model ke dalam cache
        return tf.keras.models.load_model(model_name, custom_objects=custom_objects)
    except:
        return None

# --- ALUR UTAMA ---
st.set_page_config(page_title="Accent Recognition", layout="wide")

# Mendefinisikan model di tingkat utama agar tidak "not defined"
model = load_accent_model()

# Sidebar
st.sidebar.title("⚙️ Status Sistem")
if model is not None:
    st.sidebar.success("Model: Online")
else:
    st.sidebar.error("Model: Offline (File tidak ditemukan)")

# UI Utama
st.title("🎙️ Accent Recognition")
uploaded_file = st.file_uploader("Upload file audio", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    if st.button("Mulai Prediksi"):
        if model is not None:
            with st.spinner('Sedang memproses...'):
                try:
                    # 1. Preprocessing
                    input_data = prepare_audio(uploaded_file)
                    
                    # 2. Prediksi (Gunakan model yang sudah di-load)
                    prediction = model.predict(input_data)
                    
                    # 3. Menampilkan Hasil Output
                    st.divider()
                    st.subheader("📊 Hasil Analisis")
                    
                    # Contoh menampilkan label (Sesuaikan dengan kelas aksenmu)
                    labels = ['Jawa', 'Sunda', 'Batak', 'Madura'] # Contoh label
                    predicted_class = labels[np.argmax(prediction)]
                    confidence = np.max(prediction) * 100
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Aksen Terdeteksi", predicted_class)
                    col2.metric("Tingkat Keyakinan", f"{confidence:.2f}%")
                    
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat prediksi: {e}")
        else:
            st.error("Model belum dimuat. Pastikan file model_aksen.keras sudah di-upload.")
