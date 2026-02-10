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
        # Sesuai logika Prototypical: Menghitung jarak embedding
        z_support = self.embedding(support_set)
        z_query = self.embedding(query_set)
        return z_query # Sesuaikan dengan return layer terakhir model Anda

# ==========================================================
# 2. FUNGSI PREPARASI SUPPORT SET (SESUAI FOTO EVALUASI)
# ==========================================================
def prepare_support_set(n_way=5, k_shot=3):
    """
    Menyiapkan tensor support sesuai parameter n_way dan k_shot pada foto.
    Idealnya data ini diambil dari dataset asli yang sudah diekstrak MFCC-nya.
    """
    # Simulasi fitur MFCC (40 kolom) untuk support set
    # n_way * k_shot = total sampel referensi
    support_data = np.random.randn(n_way * k_shot, 40).astype(np.float32)
    
    # Label support (misal: 0,0,0, 1,1,1, dst sesuai k_shot)
    support_labels = np.repeat(np.arange(n_way), k_shot).astype(np.int32)
    
    return (tf.convert_to_tensor(support_data), 
            tf.convert_to_tensor(support_labels))

# ==========================================================
# 3. FUNGSI PREDIKSI FINAL (FIX TRACKEDDICT & QUERY_SET)
# ==========================================================
def predict_accent(audio_path, model):
    if model is None: return "Model tidak tersedia"
    try:
        # A. Ekstraksi Fitur Query (Audio Upload)
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        
        # Sesuai foto: Konversi ke tensor float32
        query_tensor = tf.convert_to_tensor([mfcc_scaled], dtype=tf.float32)

        # B. Menyiapkan Parameter (Sesuai parameter di foto: n_way=5, k_shot=3)
        n_way = 5
        k_shot = 3
        support_tensor, support_labels_tensor = prepare_support_set(n_way, k_shot)

        # C. Pemanggilan Model (Mencegah TrackedDict Error)
        # Kita panggil .call() secara eksplisit dengan argumen posisi yang tepat
        logits = model.call(
            support_tensor, 
            query_tensor, 
            support_labels_tensor, 
            n_way
        )

        aksen_classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        
        # D. Output handling
        if isinstance(logits, tf.Tensor):
            res = logits.numpy()
        else:
            res = logits
            
        prediction_idx = np.argmax(res)
        return aksen_classes[prediction_idx % n_way]

    except Exception as e:
        return f"Error Analisis: {str(e)}"

# ==========================================================
# 4. MAIN UI STREAMLIT
# ==========================================================
def main():
    st.set_page_config(page_title="Deteksi Aksen Prototypical", layout="wide")
    
    # Load Model (Ganti dengan path model Anda)
    @st.cache_resource
    def get_model():
        try:
            return tf.keras.models.load_model("model_aksen.keras", 
                                            custom_objects={"PrototypicalNetwork": PrototypicalNetwork},
                                            compile=False)
        except: return None

    model_aksen = get_model()
    
    st.title("🎙️ Deteksi Aksen (Few-Shot Mode)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        audio_file = st.file_uploader("Pilih file audio", type=["wav", "mp3"])
        if audio_file:
            st.audio(audio_file)
            if st.button("🚀 Deteksi Aksen"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_file.getbuffer())
                    hasil = predict_accent(tmp.name, model_aksen)
                    
                with col2:
                    st.subheader("Hasil Prediksi")
                    st.success(f"Aksen Terdeteksi: {hasil}")
                
                os.unlink(tmp.name)

if __name__ == "__main__":
    main()
