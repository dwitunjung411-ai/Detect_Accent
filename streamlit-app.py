import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# 1. DEFINISI CLASS PROTOTYPICAL NETWORK (ANTI-ERROR)
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, x, training=False):
        # Menangani TrackedDict: menembus pembungkus internal Keras
        emb = self.embedding
        if isinstance(emb, dict):
            emb = emb.get('embedding', list(emb.values())[0])
        
        # Eksekusi embedding model secara langsung
        return emb(x, training=training) if callable(emb) else x

    def get_config(self):
        config = super().get_config()
        # Mengamankan serialisasi layer embedding
        config.update({"embedding_model": tf.keras.layers.serialize(self.embedding)})
        return config

# ==========================================================
# 2. FUNGSI LOAD DATA & RESOURCE
# ==========================================================
@st.cache_resource
def load_resources():
    model_name = "model_aksen.keras"
    # Menggunakan nama model sesuai instruksi koreksi user
    if os.path.exists(model_name):
        custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
        model = tf.keras.models.load_model(model_name, custom_objects=custom_objects, compile=False)
    else:
        model = None

    # Load Prototypes (Sidik jari aksen yang sudah dihitung di Colab)
    if os.path.exists("prototypes.npy"):
        prototypes = np.load("prototypes.npy")
    else:
        prototypes = None

    return model, prototypes

@st.cache_data
def load_metadata_df():
    return pd.read_csv("metadata.csv") if os.path.exists("metadata.csv") else None

# ==========================================================
# 3. LOGIKA PREDIKSI (DISTANCE-BASED CLASSIFICATION)
# ==========================================================
def predict_accent(audio_path, model, prototypes, labels):
    if model is None or prototypes is None:
        return "Resource tidak lengkap", None

    try:
        # 1. Preprocessing Audio (MFCC 40)
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        # Global Average Pooling manual agar dimensi sinkron (128,)
        mfcc_scaled = np.mean(mfcc.T, axis=0).reshape(1, -1)

        # 2. Ekstraksi Embedding (Inference)
        # Mencoba berbagai metode akses model untuk menghindari 'not callable'
        try:
            query_embedding = model.predict(mfcc_scaled, verbose=0)
        except:
            query_embedding = model(tf.convert_to_tensor(mfcc_scaled, dtype=tf.float32)).numpy()

        # 3. Klasifikasi Jarak Euclidean
        # Membandingkan query_embedding dengan matriks prototypes (5, 128)
        distances = np.linalg.norm(prototypes - query_embedding.flatten(), axis=1)
        idx = np.argmin(distances)
        
        # Hitung skor keyakinan (Confidence)
        confidence = tf.nn.softmax(-distances).numpy()
        
        return labels[idx], confidence
    except Exception as e:
        return f"Error: {str(e)}", None

# ==========================================================
# 4. MAIN UI
# ==========================================================
def main():
    st.set_page_config(page_title="Accent Detection System", page_icon="🎙️", layout="wide")
    
    model_aksen, class_prototypes = load_resources()
    df_metadata = load_metadata_df()
    
    # Daftar aksen sesuai urutan di metadata/prototypes
    aksen_classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]

    st.title("🎙️ Sistem Deteksi Aksen Indonesia")
    st.write("Klasifikasi aksen menggunakan *Prototypical Network* dan metadata referensi.")
    st.divider()

    with st.sidebar:
        st.header("🛸 Status Resource")
        if model_aksen: st.success("🤖 Model Ready")
        else: st.error("🚫 Model Missing")
        
        if class_prototypes is not None: st.success("🎯 Prototypes Ready")
        else: st.warning("⚠️ Prototypes Missing")
        
        st.divider()
        st.caption("Skripsi Project - 2026")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📥 Input Audio Lokal")
        audio_file = st.file_uploader("Upload file WAV dari laptop", type=["wav"])

        if audio_file:
            st.audio(audio_file)
            if st.button("🚀 Deteksi Aksen", type="primary", use_container_width=True):
                with st.spinner("Mengekstrak ciri suara..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_file.getbuffer())
                        tmp_path = tmp.name

                    hasil, scores = predict_accent(tmp_path, model_aksen, class_prototypes, aksen_classes)

                    with col2:
                        st.subheader("📊 Hasil Analisis")
                        with st.container(border=True):
                            st.markdown("#### Aksen Terdeteksi:")
                            st.info(f"**{hasil}**")
                            
                            if scores is not None:
                                conf_df = pd.DataFrame({'Confidence': scores}, index=aksen_classes)
                                st.bar_chart(conf_df)

                        # Pencarian info pembicara di metadata.csv berdasarkan nama file
                        st.subheader("💎 Info Metadata")
                        if df_metadata is not None:
                            match = df_metadata[df_metadata['file_name'] == audio_file.name]
                            if not match.empty:
                                info = match.iloc[0]
                                st.markdown(f"🎂 **Usia:** {info['usia']} Tahun")
                                st.markdown(f"🚻 **Gender:** {info['gender']}")
                                st.markdown(f"🗺️ **Asal:** {info['provinsi']}")
                            else:
                                st.warning("🕵️ File ini tidak tercatat di metadata.csv")
                    
                    os.unlink(tmp_path)

if __name__ == "__main__":
    main()
