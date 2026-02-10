import io
from pathlib import Path

import numpy as np
import streamlit as st
import joblib
import soundfile as sf
import librosa
import tensorflow as tf
from tensorflow.keras import layers


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Accent Recognition (Few-Shot)", page_icon="🎙️", layout="wide")

SR_DEFAULT = 22050
N_MFCC = 40
MAX_LEN = 174
N_FFT = 2048
HOP_LENGTH = 512

# Nama file yang diharapkan (sesuaikan jika berbeda)
EMBEDDING_MODEL_KERAS = "model_aksen.keras"
PREPROCESS_FILE = "preprocess.joblib"

# Batas UI agar tidak terlalu berat di Streamlit Cloud
MAX_N_WAY = 5
MAX_K_SHOT = 5


# =========================================================
# MODEL DEFINITIONS
# =========================================================
def build_embedding_model(input_shape):
    """
    Mengikuti versi yang digunakan di notebook (lebih ringkas):
    Conv2D(128) -> MaxPool -> GAP -> Dense(256) -> Dropout -> Dense(128)
    """
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
    ], name="embedding_model")
    return model


@st.cache_resource
def load_preprocess():
    """Load preprocessing objects (scaler, encoder)"""
    p = Path(PREPROCESS_FILE)
    if not p.exists():
        return None
    
    try:
        obj = joblib.load(p)
        # Validasi minimal
        if "scaler_usia" not in obj or "ohe" not in obj:
            st.warning("preprocess.joblib tidak lengkap (scaler_usia/ohe missing)")
            return None
        return obj
    except Exception as e:
        st.warning(f"Error loading preprocess.joblib: {e}")
        return None


@st.cache_resource
def load_embedding_model(expected_input_shape, uploaded_model=None):
    """
    Load embedding model dengan 3 opsi:
    1. Model yang diupload via UI
    2. Model lokal (model_aksen.keras)
    3. Build model baru (fallback jika tidak ada model)
    """
    # Opsi 1: Model yang diupload
    if uploaded_model is not None:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
                tmp.write(uploaded_model.getbuffer())
                tmp_path = tmp.name
            
            model = tf.keras.models.load_model(tmp_path, compile=False)
            Path(tmp_path).unlink()  # cleanup
            st.sidebar.success("✅ Model uploaded berhasil dimuat")
            return model
        except Exception as e:
            st.sidebar.error(f"❌ Error loading uploaded model: {e}")
    
    # Opsi 2: Model lokal
    p_model = Path(EMBEDDING_MODEL_KERAS)
    if p_model.exists():
        try:
            model = tf.keras.models.load_model(str(p_model), compile=False)
            st.sidebar.success(f"✅ Model dimuat dari {EMBEDDING_MODEL_KERAS}")
            return model
        except Exception as e:
            st.sidebar.warning(f"⚠️ Error loading {EMBEDDING_MODEL_KERAS}: {e}")
    
    # Opsi 3: Build model baru (fallback)
    st.sidebar.warning("⚠️ Model tidak ditemukan, membuat model baru (belum terlatih)")
    model = build_embedding_model(expected_input_shape)
    
    # Build weights dengan dummy data
    dummy = tf.zeros((1, *expected_input_shape), dtype=tf.float32)
    _ = model(dummy, training=False)
    
    return model


# =========================================================
# AUDIO + FEATURE EXTRACTION
# =========================================================
def load_audio_from_upload(uploaded_file, target_sr=SR_DEFAULT):
    """
    Load audio dari upload (wav/mp3/ogg/flac).
    Output: mono float32, sr=target_sr
    """
    raw = uploaded_file.read()
    try:
        y, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return y.astype(np.float32), sr
    except Exception:
        y, sr = librosa.load(io.BytesIO(raw), sr=target_sr, mono=True)
        return y.astype(np.float32), sr


def extract_mfcc_like_notebook(y, sr=SR_DEFAULT, n_mfcc=N_MFCC, max_len=MAX_LEN):
    """
    Ekstraksi MFCC + delta + delta2
    Output: (40, 174, 3)
    """
    y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Pad atau truncate
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
        delta = np.pad(delta, ((0, 0), (0, pad_width)), mode="constant")
        delta2 = np.pad(delta2, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]
        delta = delta[:, :max_len]
        delta2 = delta2[:, :max_len]

    features = np.stack([mfcc, delta, delta2], axis=-1).astype(np.float32)
    return features  # (40, 174, 3)


def build_x_final(audio_feat_40_174_3, preprocess_obj, usia, gender, provinsi):
    """
    Gabungkan audio features dengan metadata
    """
    if preprocess_obj is None:
        return audio_feat_40_174_3

    try:
        scaler_usia = preprocess_obj["scaler_usia"]
        ohe = preprocess_obj["ohe"]

        usia_scaled = scaler_usia.transform(np.array([[float(usia)]], dtype=np.float32))
        cat_encoded = ohe.transform(np.array([[str(gender), str(provinsi)]], dtype=object))

        X_meta = np.hstack([usia_scaled, cat_encoded]).astype(np.float32)
        meta_dim = X_meta.shape[1]

        X_meta_broadcast = np.repeat(X_meta[:, np.newaxis, np.newaxis, :], N_MFCC, axis=1)
        X_meta_broadcast = np.repeat(X_meta_broadcast, MAX_LEN, axis=2)

        X_audio = audio_feat_40_174_3[np.newaxis, ...]
        X_final = np.concatenate([X_audio, X_meta_broadcast], axis=-1).astype(np.float32)

        return X_final[0]
    except Exception as e:
        st.warning(f"Error processing metadata: {e}. Using audio only.")
        return audio_feat_40_174_3


# =========================================================
# PROTOTYPICAL INFERENCE
# =========================================================
def prototypical_predict(embedding_model, support_x, support_y, query_x, class_names):
    """
    Prototypical Network prediction
    """
    # Get embeddings
    sup_emb = embedding_model(support_x, training=False).numpy()
    qry_emb = embedding_model(query_x, training=False).numpy()

    n_way = len(class_names)
    prototypes = []
    
    for c in range(n_way):
        mask = (support_y == c)
        if not np.any(mask):
            raise ValueError(f"Tidak ada support untuk class index {c}.")
        prototypes.append(sup_emb[mask].mean(axis=0))
    
    prototypes = np.stack(prototypes, axis=0)

    # Euclidean distance -> logits
    dists = np.linalg.norm(qry_emb[:, None, :] - prototypes[None, :, :], axis=-1)
    logits = -dists

    # Softmax
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    pred_idx = probs.argmax(axis=1)
    
    return probs, pred_idx


# =========================================================
# UI
# =========================================================
st.title("🎙️ Accent Recognition (Few-Shot Learning)")
st.caption("Pipeline: MFCC+delta+delta2 → Embedding CNN → Prototypical Network")

# Sidebar - Settings
with st.sidebar:
    st.header("⚙️ Model & Preprocessing")
    
    # Upload model option
    uploaded_model = st.file_uploader(
        "Upload Model (.keras)", 
        type=["keras"],
        help="Upload model jika tidak tersedia di folder"
    )
    
    st.divider()
    
    st.header("📊 Episode Settings")
    n_way = st.slider("Jumlah kelas (n_way)", 2, MAX_N_WAY, 5)
    k_shot = st.slider("Contoh per kelas (k_shot)", 1, MAX_K_SHOT, 3)
    q_query = st.slider("Jumlah query audio", 1, 10, 1)

# Load preprocessing
preprocess = load_preprocess()

with st.sidebar:
    st.divider()
    st.header("🧾 Metadata")
    
    if preprocess is None:
        st.info("preprocess.joblib tidak ditemukan. Hanya menggunakan audio features.")
        use_meta = False
    else:
        use_meta = st.checkbox("Gunakan metadata", value=True)
        
        if use_meta:
            # Get categories if available
            gender_options = preprocess.get("gender_categories", ["L", "P"])
            prov_options = preprocess.get("provinsi_categories", ["Unknown"])

# Main UI
st.subheader("1️⃣ Support Set (k-shot per kelas)")
st.write(f"Upload **{k_shot}** contoh audio untuk setiap kelas")

support_files = []
support_labels = []
class_names = []
class_meta = []

cols = st.columns(n_way)
for c in range(n_way):
    with cols[c]:
        cname = st.text_input(
            f"Nama Kelas #{c+1}", 
            value=f"Kelas_{c+1}", 
            key=f"cname_{c}"
        )
        class_names.append(cname)

        # Metadata untuk kelas ini
        if preprocess is not None and use_meta:
            usia_c = st.number_input(
                "Usia", 
                min_value=0, 
                max_value=100, 
                value=25, 
                step=1, 
                key=f"usia_{c}"
            )
            
            gender_c = st.selectbox(
                "Gender", 
                gender_options, 
                key=f"gender_{c}"
            )
            
            prov_c = st.selectbox(
                "Provinsi", 
                prov_options, 
                key=f"prov_{c}"
            )
            
            class_meta.append((usia_c, gender_c, prov_c))
        else:
            class_meta.append((None, None, None))

        # Upload files
        files = st.file_uploader(
            f"Audio ({cname})",
            type=["wav", "mp3", "ogg", "flac"],
            accept_multiple_files=True,
            key=f"support_{c}",
        )

        if files:
            files = files[:k_shot]
            for f in files:
                support_files.append((c, f))
                support_labels.append(c)
            
            st.caption(f"✅ {len(files)}/{k_shot} files uploaded")

st.divider()

st.subheader("2️⃣ Query Set")
st.write(f"Upload **{q_query}** audio untuk diprediksi")

query_files = st.file_uploader(
    "Audio Query",
    type=["wav", "mp3", "ogg", "flac"],
    accept_multiple_files=True,
    key="query_upload"
)

if query_files:
    query_files = query_files[:q_query]
    st.caption(f"✅ {len(query_files)} query files uploaded")

# Query metadata (jika pakai metadata)
query_meta = None
if preprocess is not None and use_meta:
    st.subheader("🧾 Metadata Query")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        usia_q = st.number_input("Usia Query", 0, 100, 25, 1)
    with col2:
        gender_q = st.selectbox("Gender Query", gender_options)
    with col3:
        prov_q = st.selectbox("Provinsi Query", prov_options)
    
    query_meta = (usia_q, gender_q, prov_q)

st.divider()

# Run prediction
run = st.button("🔍 Jalankan Prediksi Few-Shot", type="primary", use_container_width=True)

if run:
    # Validasi
    expected_support = n_way * k_shot
    if len(support_files) < expected_support:
        st.error(f"❌ Support belum lengkap. Dibutuhkan {expected_support} files, saat ini: {len(support_files)}")
        st.stop()
    
    if not query_files:
        st.error("❌ Query audio belum diupload")
        st.stop()

    # Tentukan input shape
    if preprocess is not None and use_meta:
        scaler_usia = preprocess["scaler_usia"]
        ohe = preprocess["ohe"]
        dummy_usia = scaler_usia.transform(np.array([[25.0]], dtype=np.float32))
        dummy_cat = ohe.transform(np.array([["L", "Unknown"]], dtype=object))
        meta_dim = np.hstack([dummy_usia, dummy_cat]).shape[1]
        channels = 3 + meta_dim
    else:
        channels = 3

    expected_shape = (N_MFCC, MAX_LEN, channels)

    # Load model
    with st.spinner("⏳ Loading model..."):
        try:
            emb_model = load_embedding_model(expected_shape, uploaded_model)
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.stop()

    # Process support set
    with st.spinner("⏳ Processing support set..."):
        support_x_list = []
        progress = st.progress(0)
        
        for idx, (cls_idx, f) in enumerate(support_files):
            y, sr = load_audio_from_upload(f, target_sr=SR_DEFAULT)
            feat = extract_mfcc_like_notebook(y, sr=sr)

            if preprocess is not None and use_meta:
                usia_c, gender_c, prov_c = class_meta[cls_idx]
                x_final = build_x_final(feat, preprocess, usia_c, gender_c, prov_c)
            else:
                x_final = feat

            support_x_list.append(x_final)
            progress.progress((idx + 1) / len(support_files))
        
        progress.empty()

    support_x = np.stack(support_x_list, axis=0).astype(np.float32)
    support_y = np.array(support_labels, dtype=np.int32)

    # Process query set
    with st.spinner("⏳ Processing query set..."):
        query_x_list = []
        
        for f in query_files:
            y, sr = load_audio_from_upload(f, target_sr=SR_DEFAULT)
            feat = extract_mfcc_like_notebook(y, sr=sr)

            if preprocess is not None and use_meta and query_meta:
                usia_q, gender_q, prov_q = query_meta
                x_final = build_x_final(feat, preprocess, usia_q, gender_q, prov_q)
            else:
                x_final = feat

            query_x_list.append(x_final)

    query_x = np.stack(query_x_list, axis=0).astype(np.float32)

    # Prediction
    with st.spinner("🔮 Predicting..."):
        try:
            probs, pred_idx = prototypical_predict(
                emb_model, 
                support_x, 
                support_y, 
                query_x, 
                class_names
            )
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.stop()

    # Display results
    st.success("✅ Prediksi Selesai!")
    st.divider()
    st.subheader("📊 Hasil Prediksi")

    for i, f in enumerate(query_files):
        pred_name = class_names[int(pred_idx[i])]
        confidence = probs[i, int(pred_idx[i])] * 100
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### Query #{i+1}: {f.name}")
            st.audio(f)
        
        with col2:
            st.metric("Prediksi", pred_name, f"{confidence:.1f}%")
            
            # Top-k probabilities
            st.markdown("**Detail Probabilitas:**")
            topk = min(len(class_names), 5)
            order = np.argsort(probs[i])[::-1][:topk]
            
            for rank, j in enumerate(order, 1):
                prob_pct = probs[i, int(j)] * 100
                st.write(f"{rank}. {class_names[int(j)]}: {prob_pct:.2f}%")
        
        st.divider()

    st.balloons()
