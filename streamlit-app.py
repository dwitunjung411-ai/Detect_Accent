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
        # Mengembalikan embedding dari query_set
        return self.embedding(query_set)

# ==========================================================
# 2. FUNGSI PREDIKSI DENGAN DEBUGGING
# ==========================================================
def predict_accent_final(audio_path_string, model, audio_file_name, df_metadata):
    if model is None: 
        return "Model gagal dimuat. Periksa file .keras Anda.", None
    
    try:
        # A. Preprocessing Audio
        y, sr = librosa.load(audio_path_string, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0) 
        
        # B. Menyiapkan Tensors untuk Query Set
        query_tensor = tf.convert_to_tensor([mfcc_scaled], dtype=tf.float32)
        
        # Simulasi support set untuk kebutuhan fungsi call model
        n_way, k_shot = 5, 3
        support_tensor = tf.random.normal((n_way * k_shot, 40))
        support_labels_tensor = tf.constant(np.repeat(range(n_way), k_shot), dtype=tf.int32)

        # C. Ekstraksi Embedding (Inilah isi Query Set Anda)
        query_embedding = model.call(support_tensor, query_tensor, support_labels_tensor, n_way)
        query_embedding_np = query_embedding.numpy().tolist()

        # D. Sinkronisasi dengan Metadata
        if df_metadata is not None:
            # Membersihkan spasi untuk memastikan kecocokan nama file
            df_metadata['file_name'] = df_metadata['file_name'].str.strip()
            match = df_metadata[df_metadata['file_name'] == audio_file_name.strip()]
            
            if not match.empty:
                return match.iloc[0].get('provinsi', 'Aksen Terdeteksi'), query_embedding_np
            else:
                return f"File '{audio_file_name}' tidak ada di metadata.csv", query_embedding_np

        return "Aksen Terdeteksi (Metadata tidak ditemukan)", query_embedding_np

    except Exception as e:
        return f"Gagal Deteksi: {str(e)}", None

# ==========================================================
# 3. MAIN UI STREAMLIT
# ==========================================================
def main():
    st.set_page_config(page_title="Deteksi Aksen Prototypical", layout="wide")
    
    @st.cache_resource
    def load_resources():
        model = None
        # Menggunakan nama model sesuai instruksi terbaru Anda
        model_path = "model_detect_aksen.keras" 
        try:
            custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        except Exception as e:
            st.error(f"Gagal memuat model '{model_path}': {e}")
            
        df = pd.read_csv("metadata.csv") if os.path.exists("metadata.csv") else None
        if df is None:
            st.warning("File 'metadata.csv' tidak ditemukan.")
        return model, df

    model_aksen, df_metadata = load_resources()

    st.title("🎙️ Sistem Deteksi Aksen Prototypical")
    st.caption("Aplikasi ini menggunakan Few-Shot Learning untuk mendeteksi aksen pembicara.")
    st.divider()

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📥 Input Audio")
        audio_file = st.file_uploader("Upload file (.wav, .mp3)", type=["wav", "mp3"])

        if audio_file:
            st.audio(audio_file)
            if st.button("🚀 Jalankan Deteksi"):
                with st.spinner("Mengekstrak fitur dan memproses query set..."):
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
                st.write("#### 🎭 Aksen Terdeteksi:")
                st.info(f"**{st.session_state['hasil_aksen']}**")
            
            # FITUR BARU: Melihat isi Query Set
            with st.expander("🔍 Lihat Detail Query Set (Vektor Embedding)"):
                st.write("Vektor di bawah ini adalah representasi numerik dari suara Anda yang diproses oleh model:")
                st.write(np.array(st.session_state['query_set_data']))
        else:
            st.info("Silakan unggah audio dan klik tombol deteksi.")

        if audio_file and df_metadata is not None:
            st.divider()
            st.subheader("💎 Info Metadata")
            match = df_metadata[df_metadata['file_name'] == audio_file.name.strip()]
            if not match.empty:
                info = match.iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Usia", f"{info.get('usia', '-')} thn")
                c2.metric("Gender", info.get('gender', '-'))
                c3.metric("Provinsi", info.get('provinsi', '-'))

if __name__ == "__main__":
    main()
