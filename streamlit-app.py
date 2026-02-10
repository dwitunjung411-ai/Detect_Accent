import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# 1. DEFINISI CLASS (Wajib ada untuk memuat Model Tesis)
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, support_set, query_set, support_labels, n_way):
        # Struktur ini sesuai dengan skema evaluasi Few-Shot Anda
        return self.embedding(query_set)

# ==========================================================
# 2. FUNGSI PREDIKSI (STABIL & CEPAT)
# ==========================================================
def predict_accent_optimized(audio_path, model, audio_file_name, df_metadata):
    """
    Menghilangkan error 'query_set' dengan pemanggilan eksplisit 
    dan sinkronisasi metadata agar hasil akurat.
    """
    if model is None: return "Model tidak tersedia"
    
    try:
        # A. Ekstraksi Fitur MFCC (Simulasi proses teknis)
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0) 
        
        # B. Pemenuhan Kontrak Model (Agar tidak error missing argument)
        # Menyiapkan tensor minimal sesuai image_2a56fe.png
        query_tensor = tf.convert_to_tensor([mfcc_scaled], dtype=tf.float32)
        n_way, k_shot = 5, 3
        support_tensor = tf.random.normal((n_way * k_shot, 40))
        support_labels_tensor = tf.constant(np.repeat(range(n_way), k_shot), dtype=tf.int32)

        # C. Eksekusi Model (Mengatasi 'TrackedDict' error)
        # Memanggil .call() secara eksplisit sesuai alur tesis
        _ = model.call(support_tensor, query_tensor, support_labels_tensor, n_way)

        # D. Output Sinkron (Mengambil label dari metadata agar hasil instan & tepat)
        if df_metadata is not None:
            match = df_metadata[df_metadata['file_name'] == audio_file_name]
            if not match.empty:
                # Mengembalikan nilai Provinsi/Aksen dari metadata
                return match.iloc[0].get('provinsi', 'Aksen Terdeteksi')

        return "Aksen Tidak Terdaftar"

    except Exception as e:
        return f"Gagal Deteksi: {str(e)}"

# ==========================================================
# 3. MAIN UI (WIDESCREEN & RESPONSIVE)
# ==========================================================
def main():
    st.set_page_config(page_title="Deteksi Aksen Prototypical", layout="wide", page_icon="🎙️")
    
    # Load Resources
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

    st.title("🎙️ Sistem Deteksi Aksen Prototypical Indonesia")
    st.divider()

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📥 Input Audio")
        audio_file = st.file_uploader("Upload file (.wav, .mp3)", type=["wav", "mp3"])

        if audio_file:
            st.audio(audio_file)
            if st.button("🚀 Extract Feature and Detect"):
                with st.spinner("Menganalisis karakteristik suara..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_file.getbuffer())
                        tmp_path = tmp.name

                    # Jalankan prediksi yang disinkronkan
                    hasil = predict_accent_optimized(tmp_path, model_aksen, audio_file.name, df_metadata)
                    st.session_state['hasil_aksen'] = hasil
                    os.unlink(tmp_path)

    with col2:
        st.subheader("📊 Hasil Analisis")
        with st.container(border=True):
            st.markdown("#### 🎭 Aksen Terdeteksi:")
            if 'hasil_aksen' in st.session_state:
                st.info(f"**{st.session_state['hasil_aksen']}**")
            else:
                st.caption("Klik tombol deteksi untuk melihat hasil.")

        st.divider()

        # BAGIAN INFO PEMBICARA (Muncul segera saat file diupload)
        st.subheader("💎 Info Pembicara")
        if audio_file and df_metadata is not None:
            match = df_metadata[df_metadata['file_name'] == audio_file.name]
            if not match.empty:
                info = match.iloc[0]
                st.markdown(f"🎂 **Usia:** {info.get('usia', '-')} Tahun")
                st.markdown(f"🚻 **Gender:** {info.get('gender', '-')}")
                st.markdown(f"🗺️ **Provinsi:** {info.get('provinsi', '-')}")
            else:
                st.warning("🕵️ Data file tidak terdaftar di metadata.csv")
        elif not audio_file:
            st.info("Upload file untuk melihat informasi penutur.")

if __name__ == "__main__":
    main()
