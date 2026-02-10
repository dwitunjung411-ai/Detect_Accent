import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(page_title="Deteksi Aksen Indonesia", page_icon="🎙️", layout="wide")

# ==========================================================
# FUNGSI LOAD MODEL (DENGAN FIX ERROR KEYWORD)
# ==========================================================
@st.cache_resource
def load_accent_model():
    model_path = "model_aksen.keras"
    
    if not os.path.exists(model_path):
        st.error(f"❌ File '{model_path}' tidak ditemukan di direktori.")
        return None
    
    try:
        # Penanganan error 'embedding_model' atau keyword argument lainnya
        # Kita definisikan dummy class jika model mengharapkan objek kustom
        custom_objects = {
            'embedding_model': lambda **kwargs: None 
        }
        
        # Load model dengan compile=False untuk menghindari masalah optimizer
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects, 
            compile=False
        )
        return model
    except Exception as e:
        # Fallback jika cara pertama gagal
        try:
            model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
            return model
        except Exception as e2:
            st.error(f"Gagal memuat model: {str(e2)}")
            return None

# ==========================================================
# LOAD METADATA
# ==========================================================
@st.cache_data
def load_metadata_df():
    if os.path.exists("metadata.csv"):
        return pd.read_csv("metadata.csv")
    return None

# ==========================================================
# FUNGSI PREDIKSI
# ==========================================================
def predict_accent(audio_path, model):
    try:
        # 1. Load audio (sampling rate disamakan dengan saat training, misal 16kHz)
        y, sr = librosa.load(audio_path, sr=16000)
        
        # 2. Ekstraksi Feature (MFCC)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        # 3. Reshape untuk Input Model (batch_size, feature)
        X = np.expand_dims(mfcc_mean, axis=0)
        
        # 4. Predict
        pred = model.predict(X, verbose=0)
        
        # 5. Mapping Label
        classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        idx = np.argmax(pred[0])
        conf = pred[0][idx] * 100
        
        return classes[idx], conf
        
    except Exception as e:
        return f"Error: {str(e)}", 0

# ==========================================================
# INTERFACE (UI)
# ==========================================================
def main():
    st.title("🎙️ Deteksi Aksen Indonesia")
    st.markdown("Aplikasi ini mendeteksi aksen daerah berdasarkan sampel suara menggunakan Deep Learning.")
    st.divider()

    # Load resources
    model = load_accent_model()
    metadata = load_metadata_df()

    if model is None:
        st.warning("⚠️ Aplikasi berjalan tanpa model. Pastikan file model sudah diupload.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📥 Input Audio")
        audio_file = st.file_uploader("Upload file suara (.wav atau .mp3)", type=["wav", "mp3"])
        
        if audio_file:
            st.audio(audio_file)
            
            if st.button("🚀 Analisis Aksen", type="primary", use_container_width=True):
                with st.spinner("Sedang menganalisis karakteristik suara..."):
                    # Simpan file sementara
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(audio_file.getbuffer())
                        tmp_path = tmp_file.name
                    
                    # Jalankan Prediksi
                    label, confidence = predict_accent(tmp_path, model)
                    
                    # Tampilkan Hasil di Kolom 2
                    with col2:
                        st.subheader("📊 Hasil Analisis")
                        if confidence > 0:
                            st.success(f"Prediksi Aksen: **{label}**")
                            st.info(f"Tingkat Keyakinan: **{confidence:.2f}%**")
                            
                            # Cek Metadata jika ada
                            if metadata is not None:
                                match = metadata[metadata['file_name'] == audio_file.name]
                                if not match.empty:
                                    st.divider()
                                    st.markdown("**Informasi Speaker (Metadata):**")
                                    info = match.iloc[0]
                                    st.write(f"🎂 Usia: {info.get('usia', '-')} Tahun")
                                    st.write(f"🚻 Gender: {info.get('gender', '-')}")
                                    st.write(f"🗺️ Provinsi: {info.get('provinsi', '-')}")
                        else:
                            st.error(label)
                    
                    # Hapus file sementara
                    os.unlink(tmp_path)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Model: TensorFlow Keras | Versi 1.0")

if __name__ == "__main__":
    main()
