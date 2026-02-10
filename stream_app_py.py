import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# LOAD MODEL
# ==========================================================
@st.cache_resource
def load_accent_model():
    model_path = "model_aksen.keras"
    
    if not os.path.exists(model_path):
        st.sidebar.error(f"❌ File '{model_path}' tidak ditemukan")
        return None
    
    try:
        # Load model tanpa compile untuk menghindari error optimizer
        model = tf.keras.models.load_model(model_path, compile=False)
        st.sidebar.success("✅ Model loaded")
        return model
    except Exception as e:
        try:
            # Cara alternatif jika model menggunakan custom layer/safe_mode
            model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
            st.sidebar.success("✅ Model loaded (safe_mode=False)")
            return model
        except Exception as e2:
            st.sidebar.error(f"❌ Gagal load: {str(e2)[:100]}")
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
# FUNGSI PREDIKSI (PERBAIKAN QUERY_SET)
# ==========================================================
def predict_accent(audio_path, model):
    if model is None:
        return "Model tidak tersedia"
    
    try:
        # 1. Load audio (Sesuai dengan sample rate saat training)
        y, sr = librosa.load(audio_path, sr=16000)
        
        # 2. Ekstraksi MFCC
        # Pastikan n_mfcc sesuai dengan konfigurasi saat training tesis Anda
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        # 3. Reshape Input
        # Menambahkan dimensi batch agar menjadi (1, 40)
        X = np.expand_dims(mfcc_mean, axis=0) 
        
        # Konversi ke Tensor
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        
        # 4. Inforensi dengan argumen query_set
        # Ini adalah solusi untuk error "missing a required argument: 'query_set'"
        # Kita memanggil model secara langsung sebagai fungsi (call method)
        pred_tensor = model(query_set=X_tensor, training=False)
        
        # Konversi hasil kembali ke numpy array
        if hasattr(pred_tensor, "numpy"):
            pred = pred_tensor.numpy()
        else:
            pred = pred_tensor

        # 5. Mapping Label
        classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        idx = np.argmax(pred[0])
        conf = pred[0][idx] * 100
        
        return f"{classes[idx]} ({conf:.1f}%)"
        
    except Exception as e:
        return f"Error Detail: {str(e)}"

# ==========================================================
# UI STREAMLIT
# ==========================================================
st.set_page_config(page_title="Deteksi Aksen", page_icon="🎙️", layout="wide")

# Custom CSS untuk tampilan lebih bersih
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎙️ Deteksi Aksen Indonesia")
st.caption("Aplikasi Klasifikasi Aksen Regional menggunakan Prototypical Networks")
st.divider()

# Inisialisasi Model & Metadata
model = load_accent_model()
metadata = load_metadata_df()

# Layout Kolom
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📥 Input Audio")
    audio_file = st.file_uploader("Unggah file suara (.wav, .mp3)", type=["wav", "mp3"])
    
    if audio_file:
        st.audio(audio_file)
        
        if st.button("🚀 Jalankan Analisis", type="primary", use_container_width=True):
            if model:
                with st.spinner("Sedang memproses audio..."):
                    # Buat file sementara
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_file.getbuffer())
                        tmp_path = tmp.name
                    
                    # Jalankan Prediksi
                    result = predict_accent(tmp_path, model)
                    
                    # Tampilkan Hasil di Kolom Kanan
                    with col_output:
                        st.subheader("📊 Hasil Analisis")
                        if "Error" in result:
                            st.error(result)
                        else:
                            st.success(f"**Prediksi Aksen:** {result}")
                        
                        st.divider()
                        
                        # Menampilkan Metadata jika tersedia di metadata.csv
                        if metadata is not None:
                            # Cari baris yang nama filenya mirip dengan yang diupload
                            match = metadata[metadata['file_name'] == audio_file.name]
                            if not match.empty:
                                info = match.iloc[0]
                                st.info("ℹ️ **Informasi Metadata File:**")
                                st.write(f"🎂 **Usia:** {info.get('usia', '-')} Tahun")
                                st.write(f"🚻 **Gender:** {info.get('gender', '-')}")
                                st.write(f"🗺️ **Provinsi:** {info.get('provinsi', '-')}")
                            else:
                                st.warning("Metadata untuk file ini tidak ditemukan di CSV.")
                    
                    # Hapus file sementara setelah selesai
                    os.unlink(tmp_path)
            else:
                st.error("Gagal menjalankan analisis karena model tidak ter-load.")

with col_output:
    if not audio_file:
        st.info("Silakan unggah file audio di sebelah kiri untuk melihat hasil prediksi.")
