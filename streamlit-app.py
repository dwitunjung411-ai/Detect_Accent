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

# Nama file yang diharapkan
EMBEDDING_MODEL_KERAS = "model_aksen.keras"
PREPROCESS_FILE = "preprocess.joblib"

# Batas UI
MAX_N_WAY = 5
MAX_K_SHOT = 5


# =========================================================
# MODEL DEFINITIONS
# =========================================================
def build_embedding_model(input_shape):
    """
    Build simple CNN embedding model
    """
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
    ], name="embedding_model")
    return model


@st.cache_resource
def load_preprocess():
    """Load preprocessing objects"""
    p = Path(PREPROCESS_FILE)
    if not p.exists():
        return None
    
    try:
        obj = joblib.load(p)
        if "scaler_usia" not in obj or "ohe" not in obj:
            st.sidebar.warning("⚠️ preprocess.joblib tidak lengkap")
            return None
        return obj
    except Exception as e:
        st.sidebar.warning(f"⚠️ Error loading preprocess: {e}")
        return None


@st.cache_resource
def load_embedding_model(expected_input_shape, uploaded_model=None):
    """
    Load embedding model dengan multiple fallback
    """
    # Opsi 1: Uploaded model
    if uploaded_model is not None:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
                tmp.write(uploaded_model.getbuffer())
                tmp_path = tmp.name
            
            model = tf.keras.models.load_model(tmp_path, compile=False)
            Path(tmp_path).unlink()
            st.sidebar.success("✅ Model uploaded dimuat")
            return model
        except Exception as e:
            st.sidebar.error(f"❌ Error upload model: {str(e)[:100]}")
    
    # Opsi 2: Local model
    p_model = Path(EMBEDDING_MODEL_KERAS)
    if p_model.exists():
        try:
            model = tf.keras.models.load_model(str(p_model), compile=False)
            st.sidebar.success(f"✅ Model dimuat dari {EMBEDDING_MODEL_KERAS}")
            return model
        except Exception as e:
            st.sidebar.warning(f"⚠️ Error load {EMBEDDING_MODEL_KERAS}: {str(e)[:100]}")
    
    # Opsi 3: Build new model
    st.sidebar.warning("⚠️ Model tidak ada, membuat model baru (belum terlatih)")
    model = build_embedding_model(expected_input_shape)
    
    # Initialize weights
    dummy = tf.zeros((1, *expected_input_shape), dtype=tf.float32)
    _ = model(dummy, training=False)
    
    return model


# =========================================================
# AUDIO PROCESSING
# =========================================================
def load_audio_from_upload(uploaded_file, target_sr=SR_DEFAULT):
    """Load audio from uploaded file"""
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


def extract_mfcc_features(y, sr=SR_DEFAULT, n_mfcc=N_MFCC, max_len=MAX_LEN):
    """
    Extract MFCC + delta + delta2
    Output: (40, 174, 3)
    """
    y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Pad or truncate
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
    return features


def add_metadata(audio_feat, preprocess_obj, usia, gender, provinsi):
    """Add metadata to audio features"""
    if preprocess_obj is None:
        return audio_feat

    try:
        scaler_usia = preprocess_obj["scaler_usia"]
        ohe = preprocess_obj["ohe"]

        usia_scaled = scaler_usia.transform(np.array([[float(usia)]], dtype=np.float32))
        cat_encoded = ohe.transform(np.array([[str(gender), str(provinsi)]], dtype=object))

        X_meta = np.hstack([usia_scaled, cat_encoded]).astype(np.float32)
        meta_dim = X_meta.shape[1]

        X_meta_broadcast = np.repeat(X_meta[:, np.newaxis, np.newaxis, :], N_MFCC, axis=1)
        X_meta_broadcast = np.repeat(X_meta_broadcast, MAX_LEN, axis=2)

        X_audio = audio_feat[np.newaxis, ...]
        X_final = np.concatenate([X_audio, X_meta_broadcast], axis=-1).astype(np.float32)

        return X_final[0]
    except Exception as e:
        st.warning(f"⚠️ Metadata error: {e}. Using audio only.")
        return audio_feat


# =========================================================
# PROTOTYPICAL NETWORK
# =========================================================
def compute_prototypes(embeddings, labels, n_classes):
    """Compute prototype (mean) for each class"""
    prototypes = []
    for c in range(n_classes):
        mask = (labels == c)
        if not np.any(mask):
            raise ValueError(f"No support examples for class {c}")
        class_embeddings = embeddings[mask]
        prototype = class_embeddings.mean(axis=0)
        prototypes.append(prototype)
    return np.stack(prototypes, axis=0)


def euclidean_distance(x, y):
    """Compute Euclidean distance between x and y"""
    # x: (N, D), y: (M, D)
    # output: (N, M)
    return np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)


def prototypical_predict(model, support_x, support_y, query_x, n_classes):
    """
    Prototypical Network prediction
    
    Args:
        model: Embedding model
        support_x: Support features (Ns, H, W, C)
        support_y: Support labels (Ns,)
        query_x: Query features (Nq, H, W, C)
        n_classes: Number of classes
    
    Returns:
        probs: (Nq, n_classes)
        pred_idx: (Nq,)
    """
    # Get embeddings
    support_embeddings = model(support_x, training=False).numpy()
    query_embeddings = model(query_x, training=False).numpy()
    
    # Compute prototypes
    prototypes = compute_prototypes(support_embeddings, support_y, n_classes)
    
    # Compute distances
    distances = euclidean_distance(query_embeddings, prototypes)
    
    # Convert to logits (negative distance)
    logits = -distances
    
    # Softmax
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    
    # Predictions
    pred_idx = probs.argmax(axis=1)
    
    return probs, pred_idx


# =========================================================
# UI
# =========================================================
st.title("🎙️ Few-Shot Accent Recognition")
st.caption("Prototypical Network untuk deteksi aksen dari sedikit contoh")

# Sidebar
with st.sidebar:
    st.header("⚙️ Model")
    
    uploaded_model = st.file_uploader(
        "Upload Model (.keras)", 
        type=["keras"],
        help="Optional: upload jika tidak ada di folder"
    )
    
    st.divider()
    st.header("📊 Episode Config")
    
    n_way = st.slider("Jumlah kelas (n-way)", 2, MAX_N_WAY, 3)
    k_shot = st.slider("Contoh per kelas (k-shot)", 1, MAX_K_SHOT, 2)
    q_query = st.slider("Jumlah query", 1, 5, 1)

# Load preprocessing
preprocess = load_preprocess()

with st.sidebar:
    st.divider()
    st.header("🧾 Metadata")
    
    if preprocess is None:
        st.info("💡 Tidak pakai metadata (audio only)")
        use_meta = False
    else:
        use_meta = st.checkbox("Gunakan metadata", value=False)
        
        if use_meta:
            gender_opts = preprocess.get("gender_categories", ["L", "P"])
            prov_opts = preprocess.get("provinsi_categories", ["Unknown"])

# Main UI
st.subheader("1️⃣ Support Set")
st.write(f"Upload **{k_shot}** audio untuk setiap kelas")

support_data = []
class_names = []

cols = st.columns(n_way)
for c in range(n_way):
    with cols[c]:
        cname = st.text_input(
            f"Kelas #{c+1}", 
            value=f"Aksen_{c+1}", 
            key=f"cn_{c}"
        )
        class_names.append(cname)

        # Metadata per kelas
        if use_meta:
            usia = st.number_input("Usia", 18, 80, 25, key=f"u_{c}")
            gender = st.selectbox("Gender", gender_opts, key=f"g_{c}")
            prov = st.selectbox("Provinsi", prov_opts, key=f"p_{c}")
            meta = (usia, gender, prov)
        else:
            meta = None

        files = st.file_uploader(
            f"Audio {cname}",
            type=["wav", "mp3"],
            accept_multiple_files=True,
            key=f"sup_{c}",
        )

        if files:
            files = files[:k_shot]
            for f in files:
                support_data.append({
                    'class_idx': c,
                    'file': f,
                    'meta': meta
                })
            st.caption(f"✅ {len(files)}/{k_shot}")

st.divider()
st.subheader("2️⃣ Query Set")

query_files = st.file_uploader(
    f"Upload {q_query} audio query",
    type=["wav", "mp3"],
    accept_multiple_files=True,
    key="query"
)

if query_files:
    query_files = query_files[:q_query]
    st.caption(f"✅ {len(query_files)} query")

# Query metadata
if use_meta and query_files:
    st.write("**Metadata Query:**")
    c1, c2, c3 = st.columns(3)
    with c1:
        q_usia = st.number_input("Usia", 18, 80, 25, key="q_u")
    with c2:
        q_gender = st.selectbox("Gender", gender_opts, key="q_g")
    with c3:
        q_prov = st.selectbox("Provinsi", prov_opts, key="q_p")
    query_meta = (q_usia, q_gender, q_prov)
else:
    query_meta = None

st.divider()

# Predict button
if st.button("🔍 Jalankan Prediksi", type="primary", use_container_width=True):
    
    # Validations
    if len(support_data) < n_way * k_shot:
        st.error(f"❌ Support tidak lengkap: {len(support_data)}/{n_way*k_shot}")
        st.stop()
    
    if not query_files:
        st.error("❌ Upload query audio dulu")
        st.stop()
    
    # Determine input shape
    if use_meta and preprocess:
        scaler = preprocess["scaler_usia"]
        ohe = preprocess["ohe"]
        dum_u = scaler.transform([[25.0]])
        dum_c = ohe.transform([["L", "Unknown"]])
        meta_dim = np.hstack([dum_u, dum_c]).shape[1]
        channels = 3 + meta_dim
    else:
        channels = 3
    
    input_shape = (N_MFCC, MAX_LEN, channels)
    
    # Load model
    with st.spinner("⏳ Loading model..."):
        model = load_embedding_model(input_shape, uploaded_model)
    
    # Process support
    with st.spinner("⏳ Processing support..."):
        sup_x_list = []
        sup_y_list = []
        
        for item in support_data:
            y, sr = load_audio_from_upload(item['file'])
            feat = extract_mfcc_features(y, sr)
            
            if use_meta and item['meta']:
                u, g, p = item['meta']
                feat = add_metadata(feat, preprocess, u, g, p)
            
            sup_x_list.append(feat)
            sup_y_list.append(item['class_idx'])
        
        support_x = np.stack(sup_x_list).astype(np.float32)
        support_y = np.array(sup_y_list, dtype=np.int32)
    
    # Process query
    with st.spinner("⏳ Processing query..."):
        qry_x_list = []
        
        for qf in query_files:
            y, sr = load_audio_from_upload(qf)
            feat = extract_mfcc_features(y, sr)
            
            if use_meta and query_meta:
                u, g, p = query_meta
                feat = add_metadata(feat, preprocess, u, g, p)
            
            qry_x_list.append(feat)
        
        query_x = np.stack(qry_x_list).astype(np.float32)
    
    # Predict
    with st.spinner("🔮 Predicting..."):
        try:
            probs, preds = prototypical_predict(
                model, 
                support_x, 
                support_y, 
                query_x, 
                n_way
            )
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()
    
    # Results
    st.success("✅ Selesai!")
    st.divider()
    st.subheader("📊 Hasil")
    
    for i, qf in enumerate(query_files):
        pred_class = class_names[preds[i]]
        conf = probs[i, preds[i]] * 100
        
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.markdown(f"### {qf.name}")
            st.audio(qf)
        
        with c2:
            st.metric("Prediksi", pred_class, f"{conf:.1f}%")
            
            st.write("**Top Probabilities:**")
            for j in np.argsort(probs[i])[::-1]:
                p = probs[i, j] * 100
                st.write(f"- {class_names[j]}: {p:.1f}%")
        
        st.divider()
    
    st.balloons()
