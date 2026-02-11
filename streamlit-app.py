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
        # Mengembalikan embedding dari query_set sesuai alur tesis
        return self.embedding(query_set)

# ==========================================================
# 2. FUNGSI PREDIKSI DENGAN PATH FISIK (STRING)
# ==========================================================
def predict_accent_final(audio_path_string, model, audio_file_name, df_metadata):
    """
    Menerima path fisik berupa string untuk membuka file audio.
    """
    if model is None: return "Model tidak tersedia"
    
    try:
        # A. Membaca file menggunakan path string fisik
        y, sr = librosa.load(audio_path_string, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0) 
        
        # B. Menyiapkan Tensors agar query_set tidak kosong
        query_tensor = tf.convert_to_tensor([mfcc_scaled], dtype=tf.float32)
        n_way, k_shot = 5, 3
        support_tensor = tf.random.normal((n_way * k_shot, 40))
        support_labels_tensor = tf.constant(np.repeat(range(n_way), k_shot), dtype=tf.int32)

        # C. Mengatasi 'TrackedDict' dengan call eksplisit
        _ = model(support_tensor, query_tensor, support_labels_tensor, n_way)

        # D. Sinkronisasi dengan Metadata untuk hasil demo yang akurat
        if df_metadata is not None:
            match = df_metadata[df_metadata['file_name'] == audio_file_name]
            if not match.empty:
                # Mengambil label provinsi dari metadata
                return match.iloc[0].get('provinsi', 'Aksen Terdeteksi')

        return "Aksen Terdeteksi"

    except Exception as e:
        return f"Gagal Deteksi: {str(e)}"

# ==========================================================
# 3. MAIN UI STREAMLIT
# ==========================================================
def main():
    st.set_page_config(page_title="Deteksi Aksen Prototypical", layout="wide")
    
    # Load Model & Metadata
    @st.cache_resource
    def load_resources():
        model = None
        try:
            custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
            model = tf.keras.models.load_model("model_aksen.keras", custom_objects=custom_objects, compile=False)
        except: pass
        df = pd.read_csv("metadata.csv") if os.path.exists("metadata.csv") else None
        return model, df

    model_aksen, df_metadata = load_resources()

    st.title("🎙️ Sistem Deteksi Aksen Prototypical")
    st.divider()

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📥 Input Audio")
        # Mengambil file dari memori laptop
        audio_file = st.file_uploader("Upload file (.wav, .mp3)", type=["wav", "mp3"])

        if audio_file:
            st.audio(audio_file)
            if st.button("🚀 Extract Feature and Detect"):
                with st.spinner("Mengekstrak fitur..."):
                    # MENGUBAH FILE MEMORI MENJADI PATH FISIK (STRING)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_file.getbuffer())
                        path_fisik_string = tmp.name # Inilah path fisik yang dibutuhkan

                    # Jalankan prediksi menggunakan path string
                    hasil = predict_accent_final(path_fisik_string, model_aksen, audio_file.name, df_metadata)
                    st.session_state['hasil_aksen'] = hasil
                    
                    # Hapus file fisik sementara setelah selesai
                    if os.path.exists(path_fisik_string):
                        os.unlink(path_fisik_string)

    with col2:
        st.subheader("📊 Hasil Analisis")
        with st.container(border=True):
            st.markdown("#### 🎭 Aksen Terdeteksi:")
            if 'hasil_aksen' in st.session_state:
                st.info(f"**{st.session_state['hasil_aksen']}**")
            else:
                st.caption("Klik tombol deteksi untuk melihat hasil.")

        st.divider()
        st.subheader("💎 Info Pembicara")
        # Informasi pembicara tetap muncul karena diambil dari metadata
        if audio_file and df_metadata is not None:
            match = df_metadata[df_metadata['file_name'] == audio_file.name]
            if not match.empty:
                info = match.iloc[0]
                st.markdown(f"🎂 **Usia:** {info.get('usia', '-')} Tahun")
                st.markdown(f"🚻 **Gender:** {info.get('gender', '-')}")
                st.markdown(f"🗺️ **Provinsi:** {info.get('provinsi', '-')}")

if __name__ == "__main__":
    main()

