import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_model": tf.keras.layers.serialize(self.embedding)
            }
        )
        return config


# ==========================================================
# 2. FUNGSI LOAD DATA
# ==========================================================
@st.cache_resource
def load_accent_model():
    model_name = "model_aksen.keras"
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, model_name)

    if os.path.exists(model_path):
        try:
            custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
            # Load tanpa compile untuk stabilitas
            model = tf.keras.models.load_model(
                model_path, custom_objects=custom_objects, compile=False
            )
            return model
        except Exception as e:
            return None
    return None


@st.cache_data
def load_metadata_df():
    csv_path = "metadata.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


# ==========================================================
# 3. FUNGSI PREDIKSI (PERBAIKAN ERROR QUERY_SET)
# ==========================================================
def predict_accent(audio_path, model):
    try:
        # 1. Load audio dan ekstraksi feature seperti biasa
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        query_set = np.expand_dims(mfcc_mean, axis=0) 
        support_set = np.zeros_like(query_set) 

        pred = model.predict({
            'query_set': query_set,
            'support_set': support_set
        }, verbose=0)
        
        classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        idx = np.argmax(pred[0])
        conf = tf.nn.softmax(pred[0])[idx] * 100
        
        return classes[idx], conf
    except Exception as e:
        return f"Error Analisis: {str(e)}", 0


# ==========================================================
# 4. MAIN UI (WIDE LAYOUT & NEW ICONS)
# ==========================================================
def main():
    # Set layout lebar agar tidak sempit
    st.set_page_config(page_title="Deteksi Aksen Prototypical", page_icon="🎙️", layout="wide")

    model_aksen = load_accent_model()
    df_metadata = load_metadata_df()

    st.title("🎙️ Sistem Deteksi Aksen Prototypical Indonesia")
    st.write("Aplikasi berbasis *Few-Shot Learning* untuk klasifikasi aksen daerah.")
    st.divider()

    with st.sidebar:
        st.header("🛸 Status Sistem")
        if model_aksen:
            st.success("🤖 Model: Terhubung")
        else:
            st.error("🚫 Model: Terputus")

        if df_metadata is not None:
            st.success("📁 Metadata: Siap")
        else:
            st.warning("⚠️ Metadata: Kosong")

        st.divider()
        st.caption("Skripsi Project - 2026")

    # Pembagian kolom agar lebar
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📥 Input Audio")
        audio_file = st.file_uploader("Upload file (.wav, .mp3)", type=["wav", "mp3"])

        if audio_file:
            st.audio(audio_file)
            # Tombol diperlebar agar proporsional
            if st.button(
                "🚀 Extract Feature and Detect", type="primary", use_container_width=True
            ):
                if model_aksen:
                    with st.spinner("Menganalisis karakteristik suara..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(audio_file.getbuffer())
                            tmp_path = tmp.name

                        hasil_aksen = predict_accent(tmp_path, model_aksen)

                        # Pencarian metadata
                        user_info = None
                        if df_metadata is not None:
                            match = df_metadata[df_metadata['file_name'] == audio_file.name]
                            if not match.empty:
                                user_info = match.iloc[0].to_dict()

                        with col2:
                            st.subheader("📊 Hasil Analisis")
                            # Gunakan container agar lebih rapi
                            with st.container(border=True):
                                st.markdown(f"#### 🎭 Aksen Terdeteksi:")
                                st.info(f"**{hasil_aksen}**")

                            st.divider()
                            st.subheader("💎 Info Pembicara")
                            if user_info:
                                # Variasi emoticon baru
                                st.markdown(f"🎂 **Usia:** {user_info.get('usia', '-')} Tahun")
                                st.markdown(f"🚻 **Gender:** {user_info.get('gender', '-')}")
                                st.markdown(f"🗺️ **Provinsi:** {user_info.get('provinsi', '-')}")
                            else:
                                st.warning("🕵️ Data file tidak terdaftar di metadata.csv")

                        os.unlink(tmp_path)
                else:
                    st.error("Gagal memproses: Model tidak ditemukan.")


if __name__ == "__main__":
    main()

