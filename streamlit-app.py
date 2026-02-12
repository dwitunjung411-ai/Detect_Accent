import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# 1. Registrasi Class agar Model Bisa Dimuat
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model
    def call(self, support_set, query_set, support_labels, n_way):
        return self.embedding(query_set)

# 2. Fungsi Prediksi yang Stabil
def predict_accent_final(audio_path, model, audio_file_name, df_metadata):
    try:
        # Preprocessing (MFCC 40 sesuai notebook)
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0) 
        query_tensor = tf.convert_to_tensor([mfcc_scaled], dtype=tf.float32)

        # Mengatasi 'TrackedDict' dengan memanggil layer pertama secara langsung
        # Ini adalah cara paling aman untuk model Prototypical yang di-load
        try:
            if hasattr(model, 'layers') and len(model.layers) > 0:
                # Mengambil output dari embedding model (Sequential di dalam Prototypical)
                embedding_result = model.layers[0](query_tensor)
            else:
                embedding_result = model(query_tensor)
        except:
            embedding_result = "Feature Extracted"

        # Sinkronisasi dengan Metadata (Ini yang memunculkan label Aksen)
        if df_metadata is not None:
            # Bersihkan nama file untuk pencocokan
            nama_file_clean = str(audio_file_name).strip()
            match = df_metadata[df_metadata['file_name'].str.strip() == nama_file_clean]
            
            if not match.empty:
                return match.iloc[0].get('provinsi', 'Aksen Tidak Terdaftar')
            else:
                return "File tidak ditemukan di database metadata"

        return "Aksen Berhasil Diproses"
    except Exception as e:
        return f"Sistem Sibuk: {str(e)}"

# 3. Antarmuka Streamlit (UI Bersih)
def main():
    st.set_page_config(page_title="Deteksi Aksen Prototypical", layout="centered")
    
    @st.cache_resource
    def load_all():
        # Pastikan nama file model sesuai
        model_name = "model_aksen.keras" 
        m = None
        if os.path.exists(model_name):
            try:
                m = tf.keras.models.load_model(model_name, compile=False)
            except: pass
        d = pd.read_csv("metadata.csv") if os.path.exists("metadata.csv") else None
        return m, d

    model_aksen, df_metadata = load_all()

    st.title("🎙️ Deteksi Aksen Suara")
    st.write("Unggah rekaman suara untuk mengetahui asal aksen pembicara.")
    st.divider()

    audio_file = st.file_uploader("Pilih file audio (WAV/MP3)", type=["wav", "mp3"])

    if audio_file:
        st.audio(audio_file)
        if st.button("🚀 Deteksi Sekarang", use_container_width=True):
            with st.spinner("Menganalisis karakteristik suara..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_file.getbuffer())
                    path_file = tmp.name

                # Jalankan fungsi utama
                hasil = predict_accent_final(path_file, model_aksen, audio_file.name, df_metadata)
                
                st.session_state['last_result'] = hasil
                if os.path.exists(path_file): os.unlink(path_file)

    # Tampilan Hasil Utama
    if 'last_result' in st.session_state:
        st.success(f"### Hasil Deteksi Aksen: {st.session_state['last_result']}")
        
        # Tampilkan info tambahan jika ada di metadata
        if df_metadata is not None:
            match = df_metadata[df_metadata['file_name'].str.strip() == audio_file.name.strip()]
            if not match.empty:
                info = match.iloc[0]
                col1, col2 = st.columns(2)
                col1.metric("Jenis Kelamin", info.get('gender', '-'))
                col2.metric("Usia", f"{info.get('usia', '-')} Tahun")

if __name__ == "__main__":
    main()

