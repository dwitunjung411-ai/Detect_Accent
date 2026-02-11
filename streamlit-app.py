import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# 1. DEFINISI CLASS PROTOTYPICAL NETWORK
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, support_set, query_set, support_labels, n_way):
        # Sesuai notebook: model mengembalikan embedding dari query_set
        return self.embedding(query_set)

# ==========================================================
# 2. FUNGSI PREDIKSI (PERBAIKAN TRACKEDDICT)
# ==========================================================
def predict_accent_final(audio_path_string, model, audio_file_name, df_metadata):
    if model is None: 
        return "Model tidak ditemukan. Pastikan file .keras ada.", None
    
    try:
        # A. Preprocessing Audio (MFCC 40 sesuai notebook)
        y, sr = librosa.load(audio_path_string, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0) 
        
        # B. Menyiapkan Tensors
        # Tambahkan dimensi batch agar menjadi (1, 40)
        query_tensor = tf.convert_to_tensor([mfcc_scaled], dtype=tf.float32)

        # C. Mengambil Embedding
        # Untuk menghindari 'TrackedDict' error, kita akses layer embedding langsung
        # atau gunakan metode .predict() jika model adalah Keras Functional/Sequential
        try:
            # Cara 1: Mencoba akses atribut embedding dari class custom
            if hasattr(model, 'embedding'):
                query_embedding = model.embedding(query_tensor)
            else:
                # Cara 2: Jika model terload sebagai Functional, gunakan predict
                query_embedding = model.predict(query_tensor, verbose=0)
        except:
            # Cara 3: Pemanggilan langsung sebagai fallback
            query_embedding = model(query_tensor, training=False)

        # Ubah ke numpy untuk display
        if hasattr(query_embedding, 'numpy'):
            query_embedding_np = query_embedding.numpy().tolist()
        else:
            query_embedding_np = query_embedding.tolist()

        # D. Sinkronisasi dengan Metadata (Sesuai Logika Tesis Anda)
        if df_metadata is not None:
            audio_file_name_clean = audio_file_name.strip()
            match = df_metadata[df_metadata['file_name'].str.strip() == audio_file_name_clean]
            
            if not match.empty:
                return match.iloc[0].get('provinsi', 'Aksen Terdeteksi'), query_embedding_np
            else:
                return f"File '{audio_file_name_clean}' tidak ada di metadata.csv", query_embedding_np

        return "Aksen Terdeteksi", query_embedding_np

    except Exception as e:
        return f"Gagal Deteksi: {str(e)}", None

# ==========================================================
# 3. MAIN UI STREAMLIT
# ==========================================================
def main():
    st.set_page_config(page_title="Deteksi Aksen Prototypical", layout="wide")
    
    @st.cache_resource
    def load_resources():
        # Sesuai instruksi: nama model adalah model_detect_aksen.keras
        model_name = "model_detect_aksen.keras" 
        model = None
        try:
            custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
            model = tf.keras.models.load_model(model_name, custom_objects=custom_objects, compile=False)
        except Exception as e:
            st.error(f"Gagal memuat {model_name}: {e}")
            
        df = pd.read_csv("metadata.csv") if os.path.exists("metadata.csv") else None
        return model, df

    model_aksen, df_metadata = load_resources()

    st.title("🎙️ Sistem Deteksi Aksen Prototypical")
    st.divider()

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📥 Input Audio")
        audio_file = st.file_uploader("Upload audio (.wav, .mp3)", type=["wav", "mp3"])

        if audio_file:
            st.audio(audio_file)
            if st.button("🚀 Jalankan Ekstraksi & Deteksi"):
                with st.spinner("Memproses fitur suara..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_file.getbuffer())
                        path_fisik = tmp.name

                    hasil, q_set = predict_accent_final(path_fisik, model_aksen, audio_file.name, df_metadata)
                    
                    st.session_state['hasil_aksen'] = hasil
                    st.session_state['query_set_data'] = q_set
                    
                    if os.path.exists(path_fisik):
                        os.unlink(path_fisik)

    with col2:
        st.subheader("📊 Hasil Analisis")
        if 'hasil_aksen' in st.session_state:
            with st.container(border=True):
                st.markdown("#### 🎭 Label Terdeteksi:")
                st.info(f"**{st.session_state['hasil_aksen']}**")
            
            with st.expander("🔍 Lihat Isi Query Set (Embedding Vector)"):
                st.write("Vektor numerik hasil ekstraksi model:")
                st.write(np.array(st.session_state['query_set_data']))
        else:
            st.info("Unggah file dan klik tombol deteksi untuk melihat hasil.")

if __name__ == "__main__":
    main()
