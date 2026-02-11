import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf
import pickle
import random

# ==========================================================
# CLASS PROTOTYPICAL NETWORK (SESUAI NOTEBOOK)
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model
    
    def call(self, support_set, query_set, support_labels, n_way, training=None):
        """
        Args:
            support_set: (n_support, height, width, channels)
            query_set: (n_query, height, width, channels)
            support_labels: (n_support,)
            n_way: jumlah kelas
        Returns:
            logits: (n_query, n_way)
        """
        # Embedding untuk support dan query
        support_embeddings = self.embedding(support_set, training=training)  # (n_support, embed_dim)
        query_embeddings = self.embedding(query_set, training=training)      # (n_query, embed_dim)
        
        # Hitung prototype untuk setiap kelas
        prototypes = []
        for i in range(n_way):
            # Ambil embedding dari kelas i
            class_embeddings = tf.boolean_mask(
                support_embeddings,
                tf.equal(support_labels, i)
            )
            # Prototype = rata-rata embedding kelas i
            prototype = tf.reduce_mean(class_embeddings, axis=0)
            prototypes.append(prototype)
        
        prototypes = tf.stack(prototypes)  # (n_way, embed_dim)
        
        # Hitung jarak euclidean antara query dan prototypes
        # query_embeddings: (n_query, embed_dim)
        # prototypes: (n_way, embed_dim)
        
        # Expand dimensions untuk broadcasting
        query_expanded = tf.expand_dims(query_embeddings, 1)  # (n_query, 1, embed_dim)
        prototypes_expanded = tf.expand_dims(prototypes, 0)   # (1, n_way, embed_dim)
        
        # Hitung jarak euclidean
        distances = tf.reduce_sum(
            tf.square(query_expanded - prototypes_expanded),
            axis=-1
        )  # (n_query, n_way)
        
        # Konversi jarak ke logits (negative distance)
        logits = -distances
        
        return logits
    
    def get_config(self):
        config = super().get_config()
        if self.embedding is not None:
            config.update({"embedding_model": tf.keras.layers.serialize(self.embedding)})
        return config

# ==========================================================
# FUNGSI HELPER
# ==========================================================
def extract_mfcc(audio_path, sr=22050, n_mfcc=40, max_len=174):
    """Extract MFCC 3-channel seperti di notebook"""
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=10)
        
        # MFCC utama
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Delta
        mfcc_delta = librosa.feature.delta(mfcc)
        
        # Delta-delta
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Padding/truncating
        def pad_or_truncate(arr, max_len):
            if arr.shape[1] < max_len:
                pad_width = max_len - arr.shape[1]
                arr = np.pad(arr, ((0, 0), (0, pad_width)), mode='constant')
            else:
                arr = arr[:, :max_len]
            return arr
        
        mfcc = pad_or_truncate(mfcc, max_len)
        mfcc_delta = pad_or_truncate(mfcc_delta, max_len)
        mfcc_delta2 = pad_or_truncate(mfcc_delta2, max_len)
        
        # Stack menjadi 3 channels: (n_mfcc, max_len, 3)
        mfcc_3channel = np.stack([mfcc, mfcc_delta, mfcc_delta2], axis=-1)
        
        return mfcc_3channel.astype(np.float32)
        
    except Exception as e:
        print(f"Error extracting MFCC: {e}")
        return None

def create_episode(data, labels, n_way=5, k_shot=5, q_query=5):
    """Buat episode untuk few-shot learning"""
    unique_labels = np.unique(labels)
    selected_labels = random.sample(list(unique_labels), n_way)
    
    support_set = []
    query_set = []
    support_labels = []
    query_labels = []
    
    for label in selected_labels:
        indices = np.where(labels == label)[0]
        sampled_indices = random.sample(list(indices), k_shot + q_query)
        
        support_indices = sampled_indices[:k_shot]
        query_indices = sampled_indices[k_shot:]
        
        support_set.append(data[support_indices])
        query_set.append(data[query_indices])
        
        support_labels.extend([label] * k_shot)
        query_labels.extend([label] * q_query)
    
    support_set = np.vstack(support_set)
    query_set = np.vstack(query_set)
    support_labels = np.array(support_labels)
    query_labels = np.array(query_labels)
    
    return support_set, query_set, support_labels, query_labels

# ==========================================================
# LOAD RESOURCES
# ==========================================================
@st.cache_resource
def load_model_and_support():
    """Load model dan support set"""
    try:
        # Load model
        custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
        model = tf.keras.models.load_model(
            "model_aksen.keras",
            custom_objects=custom_objects,
            compile=False
        )
        
        # Load support set (PENTING!)
        support_set = np.load('support_set.npy')
        support_labels = np.load('support_labels.npy')
        
        st.sidebar.success("✅ Model & Support Set loaded")
        return model, support_set, support_labels
        
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")
        return None, None, None

@st.cache_data
def load_metadata():
    """Load metadata CSV"""
    if os.path.exists("metadata.csv"):
        return pd.read_csv("metadata.csv")
    return None

@st.cache_data
def load_label_encoder():
    """Load label encoder jika ada"""
    try:
        with open('label_encoder.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        # Jika tidak ada, buat manual
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.classes_ = np.array(['Betawi', 'Jawa Tengah', 'Jawa Timur', 'Sunda', 'Yogyakarta'])
        return le

# ==========================================================
# PREDIKSI (SESUAI NOTEBOOK)
# ==========================================================
def predict_accent(audio_path, model, support_set, support_labels, n_way, label_encoder):
    """
    Prediksi aksen menggunakan Prototypical Network
    Sesuai dengan fungsi detect_accent_from_audio di notebook
    """
    try:
        # 1. Extract MFCC dari audio query
        mfcc_feat = extract_mfcc(audio_path)
        
        if mfcc_feat is None:
            return "❌ Error extracting features"
        
        # 2. Expand batch dimension
        query_features = np.expand_dims(mfcc_feat, axis=0).astype(np.float32)
        
        # 3. Convert to tensors
        support_tensor = tf.convert_to_tensor(support_set, dtype=tf.float32)
        query_tensor = tf.convert_to_tensor(query_features, dtype=tf.float32)
        support_labels_tensor = tf.convert_to_tensor(support_labels, dtype=tf.int32)
        
        # 4. Forward pass ke Prototypical Network
        logits = model.call(
            support_tensor,
            query_tensor,
            support_labels_tensor,
            n_way
        )
        
        # 5. Get prediction
        pred_index = tf.argmax(logits, axis=1).numpy()[0]
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]
        
        # 6. Convert index ke label
        pred_label = label_encoder.inverse_transform([pred_index])[0]
        confidence = probs[pred_index] * 100
        
        # 7. Detail probabilitas
        detail_lines = []
        for i, (cls, prob) in enumerate(zip(label_encoder.classes_, probs)):
            marker = "👉 " if i == pred_index else "   "
            detail_lines.append(f"{marker}{cls}: {prob*100:.2f}%")
        
        detail = "\n".join(detail_lines)
        
        result = f"{pred_label} ({confidence:.1f}%)\n\n📊 Detail Probabilitas:\n{detail}"
        
        return result
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ==========================================================
# STREAMLIT UI
# ==========================================================
st.set_page_config(
    page_title="Deteksi Aksen Indonesia",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Sistem Deteksi Aksen Indonesia (Few-Shot Learning)")
st.write("Aplikasi berbasis *Prototypical Network* untuk klasifikasi aksen daerah.")
st.divider()

# Load resources
model, support_set, support_labels = load_model_and_support()
metadata = load_metadata()
label_encoder = load_label_encoder()

# Sidebar info
with st.sidebar:
    st.header("🛸 Status Sistem")
    
    if model is not None:
        st.success("🤖 Model: Aktif")
    if support_set is not None:
        st.info(f"📦 Support Set: {support_set.shape[0]} samples")
    if metadata is not None:
        st.info(f"📁 Metadata: {len(metadata)} records")
    
    st.divider()
    st.caption("🎓 Few-Shot Learning Project - 2026")

# Main layout
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📥 Input Audio")
    
    audio_file = st.file_uploader(
        "Upload file audio (.wav, .mp3)",
        type=["wav", "mp3"]
    )
    
    if audio_file:
        st.audio(audio_file)
        
        if st.button("🚀 Analisis Aksen", type="primary", use_container_width=True):
            if model is not None and support_set is not None:
                with st.spinner("🔍 Menganalisis karakteristik suara..."):
                    # Save temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_file.getbuffer())
                        tmp_path = tmp.name
                    
                    # Predict dengan n_way=5 (5 kelas aksen)
                    n_way = len(label_encoder.classes_)
                    hasil = predict_accent(
                        tmp_path, 
                        model, 
                        support_set, 
                        support_labels, 
                        n_way, 
                        label_encoder
                    )
                    
                    # Get metadata
                    user_info = None
                    if metadata is not None:
                        match = metadata[metadata['file_name'] == audio_file.name]
                        if not match.empty:
                            user_info = match.iloc[0].to_dict()
                    
                    # Display results
                    with col2:
                        st.subheader("📊 Hasil Analisis")
                        
                        with st.container(border=True):
                            st.markdown("#### 🎭 Aksen Terdeteksi:")
                            if "❌" in hasil:
                                st.error(hasil)
                            else:
                                st.text(hasil)
                        
                        st.divider()
                        
                        st.subheader("💎 Info Pembicara (dari Metadata)")
                        if user_info:
                            # Mapping provinsi ke aksen
                            province_to_accent = {
                                'DKI Jakarta': 'Betawi',
                                'Jawa Barat': 'Sunda',
                                'Jawa Tengah': 'Jawa Tengah',
                                'Jawa Timur': 'Jawa Timur',
                                'Yogyakarta': 'Yogyakarta'
                            }
                            
                            actual_province = user_info.get('provinsi', '-')
                            actual_accent = province_to_accent.get(actual_province, '-')
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("🎂 Usia", f"{user_info.get('usia', '-')} Tahun")
                                st.metric("🚻 Gender", user_info.get('gender', '-'))
                            with col_b:
                                st.metric("🗺️ Provinsi", actual_province)
                                st.metric("✅ Aksen Sebenarnya", actual_accent)
                            
                            # Check if correct
                            if actual_accent != '-':
                                predicted_accent = hasil.split('(')[0].strip()
                                if actual_accent == predicted_accent:
                                    st.success("🎯 Prediksi BENAR!")
                                else:
                                    st.warning(f"⚠️ Prediksi tidak sesuai! Seharusnya: **{actual_accent}**")
                        else:
                            st.info("🕵️ File tidak terdaftar dalam metadata")
                    
                    # Cleanup
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
            else:
                st.error("⚠️ Model atau Support Set tidak tersedia")
    else:
        with col2:
            st.info("👈 Upload file audio untuk memulai analisis")

# Info tambahan
with st.expander("ℹ️ Tentang Few-Shot Learning"):
    st.write("""
    **Prototypical Network** adalah metode Few-Shot Learning yang:
    - Menggunakan **Support Set** sebagai referensi untuk setiap kelas
    - Menghitung **prototype** (centroid) dari embedding setiap kelas
    - Mengklasifikasikan query berdasarkan jarak ke prototype terdekat
    
    **Support Set** adalah sekumpulan contoh dari setiap kelas aksen yang digunakan sebagai referensi saat prediksi.
    """)
