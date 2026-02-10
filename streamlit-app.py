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
        # Memastikan embedding dipanggil dengan query_set
        return self.embedding(query_set)

# ==========================================================
# 2. FUNGSI LOAD MODEL & METADATA
# ==========================================================
@st.cache_resource
def load_accent_model():
    try:
        custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
        model = tf.keras.models.load_model("model_aksen.keras", custom_objects=custom_objects, compile=False)
        return model
    except: return None

@st.cache_data
def load_metadata_df():
    if os.path.exists("metadata.csv"):
        return pd.read_csv("metadata.csv")
    return None

# ==========================================================
# 3. FUNGSI PREDIKSI (FIXED ARGUMENTS)
# ==========================================================
def predict_accent(audio_path, model):
    if model is None: return "Model Error"
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        
        query_tensor = tf.convert_to_tensor([mfcc_scaled], dtype=tf.float32)
        
        # Sesuai foto: Menyiapkan support set (5-way, 3-shot)
        n_way = 5
        k_shot = 3
        support_tensor = tf.random.normal((n_way * k_shot, 40))
        support_labels = tf.constant(np.repeat(range(n_way), k_shot), dtype=tf.int32)

        # Memanggil .call secara eksplisit untuk menghindari TrackedDict
        logits = model.call(support_tensor, query_tensor, support_labels, n_way)
        
        aksen_classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        prediction_idx = np.argmax(logits.numpy() if hasattr(logits, 'numpy') else logits)
        return aksen_classes[prediction_idx % n_way]
    except Exception as e:
        return f"Error Analisis: {str(e)}"

# ==========================================================
# 4. MAIN UI (PERBAIKAN LAYOUT)
# ==========================================================
def main():
    st.set_page_config(page_title="Deteksi Aksen Prototypical", layout="wide")
    
    model_aksen = load_accent_model()
    df_metadata = load_metadata_df()

    st.title("🎙️ Sistem Deteksi Aksen Prototypical Indonesia")
    st.divider()

    # Layout utama dibagi menjadi dua kolom besar
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📥 Input Audio")
        audio_file = st.file_uploader("Upload file (.wav, .mp3)", type=["wav", "mp3"])

        if audio_file:
            st.audio(audio_file)
            
            # Pemicu Deteksi
            if st.button("🚀 Extract Feature and Detect"):
                with st.spinner("Menganalisis..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_file.getbuffer())
                        tmp_path = tmp.name

                    # Jalankan Prediksi Aksen
                    hasil_aksen = predict_accent(tmp_path, model_aksen)
                    
                    # Simpan hasil ke session state agar tidak hilang saat UI refresh
                    st.session_state['hasil_aksen'] = hasil_aksen
                    os.unlink(tmp_path)

    # Kolom Kanan: Menampilkan Hasil & Info Pembicara
    with col2:
        st.subheader("📊 Hasil Analisis")
        
        # 1. Box Hasil Aksen
        with st.container(border=True):
            st.markdown("#### 🎭 Aksen Terdeteksi:")
            if 'hasil_aksen' in st.session_state:
                st.info(f"**{st.session_state['hasil_aksen']}**")
            else:
                st.write("Silakan klik tombol deteksi di sebelah kiri.")

        st.divider()
        
        # 2. Box Info Pembicara (Usia, Gender, Provinsi)
        st.subheader("💎 Info Pembicara")
        if audio_file and df_metadata is not None:
            # Cari data berdasarkan nama file yang diupload
            match = df_metadata[df_metadata['file_name'] == audio_file.name]
            
            if not match.empty:
                user_info = match.iloc[0]
                st.markdown(f"🎂 **Usia:** {user_info.get('usia', '-')} Tahun")
                st.markdown(f"🚻 **Gender:** {user_info.get('gender', '-')}")
                st.markdown(f"🗺️ **Provinsi:** {user_info.get('provinsi', '-')}")
            else:
                st.warning("🕵️ Data file tidak terdaftar di metadata.csv")
        else:
            st.info("Upload file untuk melihat info pembicara.")

if __name__ == "__main__":
    main()
