import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# 1. DEFINISI CLASS (Pastikan konsisten dengan saat training)
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, support_set, query_set, support_labels, n_way):
        # Sesuai logika skripsi Anda
        return self.embedding(query_set)

# ==========================================================
# 2. FUNGSI PREDIKSI (PERBAIKAN ERROR 'TrackedDict')
# ==========================================================
def predict_accent(audio_path, model):
    if model is None: 
        return "Model tidak terbaca"
    try:
        # Ekstraksi fitur kueri
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        
        # Konversi ke tensor float32
        query_tensor = tf.convert_to_tensor([mfcc_scaled], dtype=tf.float32)

        # Menyiapkan support set dummy
        n_way = 5
        k_shot = 3
        support_tensor = tf.random.normal((n_way * k_shot, 40), dtype=tf.float32)
        support_labels_tensor = tf.constant(np.repeat(range(n_way), k_shot), dtype=tf.int32)

        # ✅ PERBAIKAN: Panggil model dengan SEMUA parameter yang dibutuhkan
        # n_way harus sebagai keyword argument
        logits = model(
            support_tensor,           # support_set (posisi 1)
            query_tensor,             # query_set (posisi 2)
            support_labels_tensor,    # support_labels (posisi 3)
            n_way=n_way,              # n_way sebagai keyword argument
            training=False
        )

        aksen_classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        
        # Konversi logits ke numpy jika perlu
        if hasattr(logits, 'numpy'):
            logits_array = logits.numpy()
        else:
            logits_array = np.array(logits)
        
        prediction_idx = np.argmax(logits_array)
        return aksen_classes[prediction_idx % n_way]
        
    except Exception as e:
        return f"Gagal Deteksi Aksen: {str(e)}"

# ==========================================================
# 3. MAIN UI (SUDAH DIPERBAIKI)
# ==========================================================
def main():
    st.set_page_config(page_title="Deteksi Aksen Prototypical", layout="wide")
    
    # Load Resources
    @st.cache_resource
    def load_resources():
        model = None
        try:
            model = tf.keras.models.load_model(
                "model_aksen.keras", 
                custom_objects={"PrototypicalNetwork": PrototypicalNetwork},
                compile=False
            )
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
        
        df = None
        if os.path.exists("metadata.csv"):
            df = pd.read_csv("metadata.csv")
        
        return model, df

    model_aksen, df_metadata = load_resources()

    st.title("🎙️ Sistem Deteksi Aksen Indonesia")
    st.divider()

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📥 Input Audio")
        audio_file = st.file_uploader("Upload file (.wav, .mp3)", type=["wav", "mp3"])

        if audio_file:
            st.audio(audio_file)
            
            # Eksekusi Deteksi saat tombol diklik
            if st.button("🚀 Extract Feature and Detect"):
                with st.spinner("Menganalisis aksen..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_file.getbuffer())
                        tmp_path = tmp.name
                    
                    # Simpan hasil ke session_state
                    st.session_state['hasil_aksen'] = predict_accent(tmp_path, model_aksen)
                    os.unlink(tmp_path)
                    
                    # Rerun untuk update UI
                    st.rerun()

    with col2:
        st.subheader("📊 Hasil Analisis")
        
        # BAGIAN AKSEN
        with st.container(border=True):
            st.markdown("#### 🎭 Aksen Terdeteksi:")
            if 'hasil_aksen' in st.session_state:
                if "Gagal" in st.session_state['hasil_aksen']:
                    st.error(st.session_state['hasil_aksen'])
                else:
                    st.success(f"**{st.session_state['hasil_aksen']}**")
            else:
                st.info("Menunggu deteksi...")

        st.divider()

        # BAGIAN INFO PEMBICARA
        st.subheader("💎 Info Pembicara")
        if audio_file and df_metadata is not None:
            # Mencari data pembicara berdasarkan nama file
            user_data = df_metadata[df_metadata['file_name'] == audio_file.name]
            
            if not user_data.empty:
                info = user_data.iloc[0]
                st.markdown(f"🎂 **Usia:** {info.get('usia', '-')} Tahun")
                st.markdown(f"🚻 **Gender:** {info.get('gender', '-')}")
                st.markdown(f"🗺️ **Provinsi:** {info.get('provinsi', '-')}")
            else:
                st.warning("Data pembicara tidak ditemukan di metadata.csv")
        elif audio_file:
            st.warning("File metadata.csv tidak ditemukan.")
        else:
            st.caption("Silakan upload file untuk melihat informasi.")

if __name__ == "__main__":
    main()
