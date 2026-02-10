import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==========================================================
# 1. DEFINISI CLASS PROTOTYPICAL NETWORK
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, support_set, query_set, support_labels, n_way):
        # Memastikan embedding dipanggil dengan query_set
        return self.embedding(query_set)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_model": tf.keras.layers.serialize(self.embedding)
            }
        )
        return config


# ==========================================================
# 2. FUNGSI LOAD DATA
# ==========================================================
@st.cache_resource
def load_accent_model():
    model_name = "model_aksen.keras"
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, model_name)

    if os.path.exists(model_path):
        try:
            custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
            # Load tanpa compile untuk stabilitas
            model = tf.keras.models.load_model(
                model_path, custom_objects=custom_objects, compile=False
            )
            return model
        except Exception as e:
            return None
    return None


@st.cache_data
def load_metadata_df():
    csv_path = "metadata.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


# ==========================================================
# 3. FUNGSI PREDIKSI (PERBAIKAN ERROR QUERY_SET)
# ==========================================================
def predict_accent(audio_path, model):
    try:
        # 1. Load audio dan ekstraksi feature seperti biasa
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        query_set = np.expand_dims(mfcc_mean, axis=0) 
        support_set = np.zeros_like(query_set) 

        pred = model.predict({
            'query_set': query_set,
            'support_set': support_set
        }, verbose=0)
        
        classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        idx = np.argmax(pred[0])
        conf = tf.nn.softmax(pred[0])[idx] * 100
        
        return classes[idx], conf
    except Exception as e:
        return f"Error Analisis: {str(e)}", 0

# =========================================================
# AUDIO + FEATURE EXTRACTION (sesuai notebook extract_mfcc)
# =========================================================
def load_audio_from_upload(uploaded_file, target_sr=SR_DEFAULT):
    """
    Load audio dari upload (wav/mp3/ogg/flac).
    - Coba soundfile dulu
    - Fallback ke librosa
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
    Replikasi fungsi extract_mfcc di notebook:
    - normalize amplitude
    - mfcc (n_mfcc=40, n_fft=2048, hop_length=512)
    - delta & delta2
    - pad/truncate ke max_len=174
    - stack channel => (40, 174, 3)
    """
    y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

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
    Meniru pipeline notebook:
    - usia -> scaler_usia
    - [gender, provinsi] -> ohe
    - gabung X_meta = [usia_scaled, cat_encoded]
    - broadcast X_meta ke (40,174,meta_dim)
    - concat jadi (40,174,3+meta_dim)
    """
    if preprocess_obj is None:
        # Tanpa preprocess: hanya audio channel (akan mismatch jika model dilatih pakai meta)
        return audio_feat_40_174_3

    scaler_usia = preprocess_obj["scaler_usia"]
    ohe = preprocess_obj["ohe"]

    usia_scaled = scaler_usia.transform(np.array([[float(usia)]], dtype=np.float32))  # (1,1)
    cat_encoded = ohe.transform(np.array([[str(gender), str(provinsi)]], dtype=object))  # (1, C)

    X_meta = np.hstack([usia_scaled, cat_encoded]).astype(np.float32)  # (1, meta_dim)
    meta_dim = X_meta.shape[1]

    X_meta_broadcast = np.repeat(X_meta[:, np.newaxis, np.newaxis, :], N_MFCC, axis=1)
    X_meta_broadcast = np.repeat(X_meta_broadcast, MAX_LEN, axis=2)  # (1,40,174,meta_dim)

    X_audio = audio_feat_40_174_3[np.newaxis, ...]  # (1,40,174,3)
    X_final = np.concatenate([X_audio, X_meta_broadcast], axis=-1).astype(np.float32)  # (1,40,174,3+meta_dim)

    return X_final[0]  # kembalikan (40,174,channels)


# =========================================================
# PROTOTYPICAL INFERENCE
# =========================================================
def prototypical_predict(embedding_model, support_x, support_y, query_x, class_names):
    """
    support_x: (Ns, H, W, C)
    support_y: (Ns,) integer 0..n_way-1
    query_x:   (Nq, H, W, C)
    class_names: list[str] panjang n_way

    Output:
    - probs (Nq, n_way)
    - pred_idx (Nq,)
    """
    # embeddings
    sup_emb = embedding_model(support_x, training=False).numpy()   # (Ns, D)
    qry_emb = embedding_model(query_x, training=False).numpy()     # (Nq, D)

    n_way = len(class_names)
    prototypes = []
    for c in range(n_way):
        mask = (support_y == c)
        if not np.any(mask):
            raise ValueError(f"Tidak ada support untuk class index {c}.")
        prototypes.append(sup_emb[mask].mean(axis=0))
    prototypes = np.stack(prototypes, axis=0)  # (n_way, D)

    # Euclidean distance -> logits = -distance
    # dist(q, p) = ||q - p||
    dists = np.linalg.norm(qry_emb[:, None, :] - prototypes[None, :, :], axis=-1)  # (Nq, n_way)
    logits = -dists

    # softmax
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    pred_idx = probs.argmax(axis=1)
    return probs, pred_idx


# =========================================================
# UI
# =========================================================
st.title("🎙️ Accent Recognition (Few-Shot Learning — Prototypical Network)")
st.caption("Sesuai pipeline notebook: MFCC+delta+delta2, metadata broadcast, embedding CNN, prototype mean + Euclidean distance.")

preprocess = None
try:
    preprocess = load_preprocess()
except Exception as e:
    st.warning(f"Preprocess tidak bisa diload: {e}")

with st.sidebar:
    st.header("⚙️ Pengaturan Episode")
    n_way = st.slider("Jumlah kelas (n_way)", 2, MAX_N_WAY, 5)
    k_shot = st.slider("Jumlah contoh per kelas (k_shot)", 1, MAX_K_SHOT, 3)
    q_query = st.slider("Jumlah query audio", 1, 10, 1)

    st.divider()
    st.header("🧾 Metadata (opsional, tapi penting jika model dilatih pakai meta)")
    if preprocess is None:
        st.info("preprocess.joblib tidak ditemukan / gagal diload. Metadata tidak dipakai (raw audio features saja).")
        use_meta = False
    else:
        use_meta = st.checkbox("Gunakan metadata (usia, gender, provinsi)", value=True)

    # opsi dropdown kalau disediakan di preprocess.joblib
    gender_options = preprocess.get("gender_categories") if preprocess else None
    prov_options = preprocess.get("provinsi_categories") if preprocess else None

st.subheader("1) Support Set (k-shot per kelas)")
st.write("Upload contoh audio per kelas. Setiap kelas butuh **k_shot** file.")

support_files = []
support_labels = []
class_names = []

# metadata per kelas (lebih realistis dibanding satu metadata untuk semua)
class_meta = []

cols = st.columns(n_way)
for c in range(n_way):
    with cols[c]:
        cname = st.text_input(f"Nama Kelas #{c+1}", value=f"Kelas_{c+1}", key=f"cname_{c}")
        class_names.append(cname)

        if preprocess is not None and use_meta:
            usia_c = st.number_input("Usia (kelas ini)", min_value=0, max_value=100, value=25, step=1, key=f"usia_{c}")
            if gender_options:
                gender_c = st.selectbox("Gender (kelas ini)", gender_options, key=f"gender_{c}")
            else:
                gender_c = st.selectbox("Gender (kelas ini)", ["L", "P"], key=f"gender_{c}")

            if prov_options:
                prov_c = st.selectbox("Provinsi (kelas ini)", prov_options, key=f"prov_{c}")
            else:
                prov_c = st.text_input("Provinsi (kelas ini)", value="Unknown", key=f"prov_{c}")

            class_meta.append((usia_c, gender_c, prov_c))
        else:
            class_meta.append((None, None, None))

        files = st.file_uploader(
            f"Upload {k_shot} audio ({cname})",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
            accept_multiple_files=True,
            key=f"support_upload_{c}",
        )

        if files:
            # batasi sesuai k_shot agar konsisten
            files = files[:k_shot]
            for f in files:
                support_files.append((c, f))
                support_labels.append(c)

st.subheader("2) Query Set")
query_files = st.file_uploader(
    f"Upload {q_query} audio query",
    type=["wav", "mp3", "ogg", "flac", "m4a"],
    accept_multiple_files=True,
)
if query_files:
    query_files = query_files[:q_query]

# tombol proses
st.divider()
run = st.button("🔍 Jalankan Prediksi Few-Shot", type="primary")

if run:
    # validasi minimal
    if len(support_files) < n_way * k_shot:
        st.error(f"Support belum lengkap. Dibutuhkan {n_way*k_shot} file (n_way*k_shot). Saat ini: {len(support_files)}.")
        st.stop()
    if not query_files or len(query_files) == 0:
        st.error("Query audio belum diupload.")
        st.stop()

    # Tentukan input shape model
    # - Jika pakai metadata, channels = 3 + meta_dim
    # - Jika tidak, channels = 3
    if preprocess is not None and use_meta:
        # hitung meta_dim dari ohe + 1 (usia)
        # gunakan dummy transform untuk tahu dimensi
        scaler_usia = preprocess["scaler_usia"]
        ohe = preprocess["ohe"]
        dummy_usia = scaler_usia.transform(np.array([[25.0]], dtype=np.float32))
        dummy_cat = ohe.transform(np.array([["L", "Unknown"]], dtype=object))
        meta_dim = np.hstack([dummy_usia, dummy_cat]).shape[1]
        channels = 3 + meta_dim
    else:
        channels = 3

    expected_shape = (N_MFCC, MAX_LEN, channels)

    # load model
    try:
        emb_model = load_embedding_model(expected_shape)
    except Exception as e:
        st.error(f"Gagal load embedding model: {e}")
        st.stop()

    # ekstraksi support
    st.info("Memproses audio support & query…")
    support_x_list = []
    for (cls_idx, f) in support_files:
        y, sr = load_audio_from_upload(f, target_sr=SR_DEFAULT)
        feat = extract_mfcc_like_notebook(y, sr=sr)

        if preprocess is not None and use_meta:
            usia_c, gender_c, prov_c = class_meta[cls_idx]
            if usia_c is None:
                # fallback (tidak semestinya terjadi jika use_meta True)
                usia_c, gender_c, prov_c = 25, "L", "Unknown"
            x_final = build_x_final(feat, preprocess, usia_c, gender_c, prov_c)  # (40,174,channels)
        else:
            x_final = feat  # (40,174,3)

        support_x_list.append(x_final)

    support_x = np.stack(support_x_list, axis=0).astype(np.float32)  # (Ns,40,174,C)
    support_y = np.array(support_labels, dtype=np.int32)             # (Ns,)

    # ekstraksi query
    query_x_list = []
    for f in query_files:
        y, sr = load_audio_from_upload(f, target_sr=SR_DEFAULT)
        feat = extract_mfcc_like_notebook(y, sr=sr)

        if preprocess is not None and use_meta:
            # Query: user masukkan metadata global (lebih masuk akal)
            st.sidebar.subheader("🧾 Metadata Query")
            usia_q = st.sidebar.number_input("Usia (query)", min_value=0, max_value=100, value=25, step=1, key="usia_query")
            if gender_options:
                gender_q = st.sidebar.selectbox("Gender (query)", gender_options, key="gender_query")
            else:
                gender_q = st.sidebar.selectbox("Gender (query)", ["L", "P"], key="gender_query")
            if prov_options:
                prov_q = st.sidebar.selectbox("Provinsi (query)", prov_options, key="prov_query")
            else:
                prov_q = st.sidebar.text_input("Provinsi (query)", value="Unknown", key="prov_query")

            x_final = build_x_final(feat, preprocess, usia_q, gender_q, prov_q)
        else:
            x_final = feat

        query_x_list.append(x_final)

    query_x = np.stack(query_x_list, axis=0).astype(np.float32)  # (Nq,40,174,C)

    # prediksi prototypical
    try:
        probs, pred_idx = prototypical_predict(emb_model, support_x, support_y, query_x, class_names)
    except Exception as e:
        st.error(f"Gagal prediksi: {e}")
        st.stop()

    # tampilkan hasil
    st.subheader("✅ Hasil Prediksi")
    for i, f in enumerate(query_files):
        pred_name = class_names[int(pred_idx[i])]
        st.markdown(f"**Query #{i+1}: {f.name} → Prediksi: _{pred_name}_**")

        # tampilkan top-k probabilitas
        topk = min(5, len(class_names))
        order = np.argsort(probs[i])[::-1][:topk]
        for j in order:
            st.write(f"- {class_names[int(j)]}: {probs[i, int(j)]:.3f}")

        st.audio(f)

    st.success("Selesai.")
