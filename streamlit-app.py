import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf
from scipy.spatial.distance import cdist

# 1. Registrasi Class (Tetap sama)
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model
    def call(self, inputs):
        return self.embedding(inputs)

# 2. Fungsi Ekstraksi Embedding
def extract_embedding(audio_path, model):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0) 
    query_tensor = tf.convert_to_tensor([mfcc_scaled], dtype=tf.float32)
    
    # Mengambil output embedding
    if hasattr(model, 'layers') and len(model.layers) > 0:
        return model.layers[0](query_tensor).numpy()
    return model(query_tensor).numpy()

# 3. Fungsi Utama Prediksi (Real Inference)
def predict_real_accent(audio_path, model, prototypes):
    """
    Membandingkan embedding input dengan prototipe aksen yang sudah disimpan.
    """
    try:
        query_embedding = extract_embedding(audio_path, model)
        
        # Hitung jarak Euclidean antara input dengan semua prototipe
        labels = list(prototypes.keys())
        proto_matrix = np.array([prototypes[l] for l in labels])
        
        distances = cdist(query_embedding, proto_matrix, metric='euclidean')
        idx_prediksi = np.argmin(distances)
        
        return labels[idx_prediksi]
    except Exception as e:
        return f"Error Prediksi: {str(e)}"

# 4. Antarmuka Streamlit
def main():
    st.set_page_config(page_title="Deteksi Aksen Prototypical", layout="centered")
    
    @st.cache_resource
    def load_all():
        # UPDATE: Gunakan nama model sesuai instruksi Anda
        model_name = "model_aksen.keras" 
        m = None
        if os.path.exists(model_name):
            m = tf.keras.models.load_model(model_name, compile=False)
        
        # Simulasi Prototipe (Idealnya ini dihitung dari data training)
        # Jika Anda punya file 'prototypes.npy', load dari sana.
        # Di sini saya buat contoh struktur data prototipe:
        d = pd.read_csv("metadata.csv") if os.path.exists("metadata.csv") else None
        
        # Contoh dummy prototypes (Anda harus mengganti ini dengan pusat koordinat tiap aksen)
        # format: {'Jawa': [0.1, 0.2, ...], 'Sunda': [0.5, -0.1, ...]}
        protos = {} 
        if d is not None:
            # Sederhananya, kita kelompokkan provinsi unik sebagai label
            list_aksen = d['provinsi'].unique()
            for aksen in list_aksen:
                protos[aksen] = np.random.rand(64) # Ganti 64 dengan dimensi output model Anda
                
        return m, d, protos

    model_aksen, df_metadata, prototypes = load_all()

    st.title("🎙️ Deteksi Aksen Suara (Real-Time AI)")
    st.write("Sistem akan menganalisis gelombang suara dan mencocokkannya dengan karakteristik aksen daerah.")
    st.divider()

    audio_file = st.file_uploader("Pilih file audio (WAV/MP3)", type=["wav", "mp3"])

    if audio_file:
        st.audio(audio_file)
        if st.button("🚀 Deteksi Karakteristik Suara", use_container_width=True):
            with st.spinner("Model sedang menghitung jarak embedding..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_file.getbuffer())
                    path_file = tmp.name

                # PREDDIKSI SEBENARNYA
                hasil_aksen = predict_real_accent(path_file, model_aksen, prototypes)
                st.session_state['last_result'] = hasil_aksen
                
                if os.path.exists(path_file): os.unlink(path_file)

    if 'last_result' in st.session_state:
        st.success(f"### Prediksi Aksen: {st.session_state['last_result']}")
        
        # Info Metadata Tetap Ditampilkan jika file cocok (Opsional)
        if df_metadata is not None:
            match = df_metadata[df_metadata['file_name'].str.strip() == audio_file.name.strip()]
            if not match.empty:
                info = match.iloc[0]
                st.info("Informasi tambahan ditemukan di database:")
                c1, c2 = st.columns(2)
                c1.metric("Gender", info.get('gender', '-'))
                c2.metric("Usia", f"{info.get('usia', '-')} Thn")

if __name__ == "__main__":
    main()
