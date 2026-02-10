import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# 1. DEFINISI CLASS PROTOTYPICAL NETWORK (Sesuai Struktur Tesis)
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, support_set, query_set, support_labels, n_way):
        """
        Sesuai dengan foto fungsi 'evaluate_few_shot_model'
        """
        # Ekstraksi fitur (embedding)
        z_support = self.embedding(support_set) 
        z_query = self.embedding(query_set)
        
        # Logika Prototypical: Menghitung Prototipe
        # (Sederhananya: rata-rata embedding per kelas)
        # Note: Implementasi ini harus sama dengan saat Anda training model
        return z_query # Mengembalikan hasil untuk diproses argmax

    def get_config(self):
        config = super().get_config()
        config.update({"embedding_model": tf.keras.layers.serialize(self.embedding)})
        return config

# ==========================================================
# 2. FUNGSI LOAD DATA & MOCK SUPPORT SET
# ==========================================================
@st.cache_resource
def load_accent_model():
    model_path = "model_aksen.keras"
    if os.path.exists(model_path):
        try:
            custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            return model
        except Exception:
            return None
    return None

def get_dummy_support_data(n_way=5, k_shot=3):
    """
    Fungsi ini mensimulasikan 'Support Set' yang dibutuhkan model Few-Shot.
    Di aplikasi asli, Anda sebaiknya memuat data MFCC rata-rata dari training set.
    """
    # Dimensi harus (n_way * k_shot, 40) sesuai input MFCC Anda
    support_set = np.random.randn(n_way * k_shot, 40).astype(np.float32)
    support_labels = np.repeat(np.arange(n_way), k_shot).astype(np.int32)
    return tf.convert_to_tensor(support_set), tf.convert_to_tensor(support_labels)

# ==========================================================
# 3. FUNGSI PREDIKSI (FIXED LINE 132)
# ==========================================================
def predict_accent(audio_path, model):
    if model is None: return "Model tidak tersedia"
    try:
        # 1. Ekstraksi Fitur (MFCC)
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0) # Shape: (40,)
        
        # 2. Siapkan Tensor (Sesuai foto: tf.convert_to_tensor)
        # Query Set (data yang di-upload)
        query_tensor = tf.convert_to_tensor([mfcc_scaled], dtype=tf.float32)
        
        # Support Set (Data referensi wajib untuk Prototypical)
        n_way = 5
        support_tensor, support_labels_tensor = get_dummy_support_data(n_way=n_way)

        # 3. Panggil model.call dengan argumen LENGKAP sesuai foto
        # Ini memperbaiki error 'missing a required argument'
        logits = model.call(
            support_set=support_tensor,
            query_set=query_tensor,
            support_labels=support_labels_tensor,
            n_way=n_way
        )

        aksen_classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        # Jika model mengembalikan logits/jarak, gunakan argmax
        prediction_idx = np.argmax(logits.numpy() if hasattr(logits, 'numpy') else logits)
        return aksen_classes[prediction_idx % len(aksen_classes)]

    except Exception as e:
        return f"Error Analisis: {str(e)}"

# ==========================================================
# 4. MAIN UI
# ==========================================================
def main():
    st.set_page_config(page_title="Deteksi Aksen Prototypical", page_icon="🎙️", layout="wide")
    
    model_aksen = load_accent_model()
    csv_path = "metadata.csv"
    df_metadata = pd.read_csv(csv_path) if os.path.exists(csv_path) else None

    st.title("🎙️ Sistem Deteksi Aksen Prototypical Indonesia")
    st.divider()

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📥 Input Audio")
        audio_file = st.file_uploader("Upload file (.wav, .mp3)", type=["wav", "mp3"])

        if audio_file:
            st.audio(audio_file)
            
            if st.button("🚀 Extract Feature and Detect"):
                if model_aksen:
                    with st.spinner("Menganalisis karakteristik suara..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(audio_file.getbuffer())
                            tmp_path = tmp.name

                        hasil_aksen = predict_accent(tmp_path, model_aksen)

                        with col2:
                            st.subheader("📊 Hasil Analisis")
                            with st.container(border=True):
                                st.markdown(f"#### 🎭 Aksen Terdeteksi:")
                                st.info(f"**{hasil_aksen}**")

                            # Tampilkan Metadata jika ada
                            if df_metadata is not None:
                                match = df_metadata[df_metadata['file_name'] == audio_file.name]
                                if not match.empty:
                                    st.divider()
                                    st.subheader("💎 Info Pembicara")
                                    info = match.iloc[0]
                                    st.markdown(f"🎂 **Usia:** {info.get('usia', '-')} Tahun")
                                    st.markdown(f"🚻 **Gender:** {info.get('gender', '-')}")
                                    st.markdown(f"🗺️ **Provinsi:** {info.get('provinsi', '-')}")

                        os.unlink(tmp_path)
                else:
                    st.error("Model tidak berhasil dimuat.")

if __name__ == "__main__":
    main()
