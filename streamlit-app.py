import streamlit as st
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import tensorflow as tf
import os
import tempfile
import random
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from pydub import AudioSegment # Needed for normalize_audio if used

# Ensure custom objects are registered for model loading
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import keras

@keras.saving.register_keras_serializable()
class PrototypicalNetwork(Model):
    def __init__(self, embedding_model, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, support_set, query_set, support_labels, n_way):
        # Hitung embedding
        support_embeddings = self.embedding(support_set)
        query_embeddings = self.embedding(query_set)

        # Hitung prototype per kelas
        prototypes = []
        for i in range(n_way):
            mask = tf.equal(support_labels, i)
            class_embeddings = tf.boolean_mask(support_embeddings, mask)
            prototype = tf.reduce_mean(class_embeddings, axis=0)
            prototypes.append(prototype)
        prototypes = tf.stack(prototypes)

        # Hitung jarak Euclidean antara query dan prototype
        distances = []
        for q in query_embeddings:
            dist = tf.norm(prototypes - q, axis=1)
            distances.append(dist)
        distances = tf.stack(distances)

        # Ubah jarak menjadi probabilitas (softmax over negative distances)
        logits = -distances
        return logits

    def get_config(self):
        config = super(PrototypicalNetwork, self).get_config()
        config.update({
            "embedding_model": keras.saving.serialize_keras_object(self.embedding)
        })
        return config

    @classmethod
    def from_config(cls, config):
        embedding_config = config.pop("embedding_model")
        embedding_model = keras.saving.deserialize_keras_object(embedding_config)
        return cls(embedding_model, **config)

# --- Data Loading and Preprocessing ---

# Use st.cache_resource to load and preprocess data only once
@st.cache_resource
def load_and_preprocess_data():
    # Define path
    path = '/content/drive/MyDrive/Voice_Skripsi_fix'

    # Load metadata
    csv_path = os.path.join(path, 'metadata.csv')
    metadata = pd.read_csv(csv_path)

    # Feature extraction function (needs to be defined locally or passed)
    def extract_mfcc_local(file_path, sr=22050, n_mfcc=40, max_len=174):
        try:
            y, sr = librosa.load(file_path, sr=sr)
            y = librosa.util.normalize(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)

            if mfcc.shape[1] < max_len:
                pad_width = max_len - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
                delta = np.pad(delta, ((0, 0), (0, pad_width)), mode='constant')
                delta2 = np.pad(delta2, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc = mfcc[:, :max_len]
                delta = delta[:, :max_len]
                delta2 = delta2[:, :max_len]

            features = np.stack([mfcc, delta, delta2], axis=-1)
            return features

        except Exception as e:
            st.error(f"Error extracting MFCC from {file_path}: {e}")
            return None

    # Extract features and metadata
    X_audio_features = []
    X_meta_raw = []
    y_text = []

    for i, row in metadata.iterrows():
        file_name = str(row['file_name'])
        file_path = os.path.join(path, file_name)

        if not os.path.exists(file_path):
            st.warning(f"File not found: {file_path}. Skipping.")
            continue

        mfcc_feat = extract_mfcc_local(file_path)
        if mfcc_feat is None:
            continue # Skip if MFCC extraction failed

        X_audio_features.append(mfcc_feat)
        X_meta_raw.append([row['usia'], row['gender'], row['provinsi']])
        y_text.append(row['label_aksen'])

    X_audio_features = np.array(X_audio_features, dtype=np.float32)
    X_meta_raw = np.array(X_meta_raw, dtype=object)
    y_text = np.array(y_text, dtype=str)

    # Label Encoding for y_text
    le_y = LabelEncoder()
    y_aksen = le_y.fit_transform(y_text)

    # Prepare target variables for multi-task learning and metadata processing
    y_usia = X_meta_raw[:, 0].astype(float)
    y_gender_raw = X_meta_raw[:, 1].astype(str)
    y_provinsi_raw = X_meta_raw[:, 2].astype(str)

    le_gender = LabelEncoder()
    le_gender.fit(y_gender_raw) # Fit the encoder
    y_gender = le_gender.transform(y_gender_raw)

    le_provinsi = LabelEncoder()
    le_provinsi.fit(y_provinsi_raw) # Fit the encoder
    y_provinsi = le_provinsi.transform(y_provinsi_raw)

    # Scale usia
    scaler_usia = StandardScaler()
    scaler_usia.fit(y_usia.reshape(-1, 1)) # Fit the scaler
    usia_scaled = scaler_usia.transform(y_usia.reshape(-1, 1))

    # One-hot for gender + provinsi
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(np.hstack([y_gender_raw.reshape(-1,1), y_provinsi_raw.reshape(-1,1)])) # Fit the OHE
    cat_encoded = ohe.transform(np.hstack([y_gender_raw.reshape(-1,1), y_provinsi_raw.reshape(-1,1)]))

    # Gabungkan metadata jadi satu
    X_meta = np.hstack([usia_scaled, cat_encoded]).astype(np.float32)

    X_meta_broadcast = np.repeat(X_meta[:, np.newaxis, np.newaxis, :],
                                 X_audio_features.shape[1], axis=1)
    X_meta_broadcast = np.repeat(X_meta_broadcast,
                                 X_audio_features.shape[2], axis=2)
    X_final = np.concatenate([X_audio_features, X_meta_broadcast], axis=-1).astype(np.float32)

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_aksen, test_size=0.2, random_state=42, stratify=y_aksen
    )
    # Return all necessary components
    return le_y, scaler_usia, le_gender, le_provinsi, ohe, X_train, y_train, extract_mfcc_local

# Load all data and preprocessing objects
le_y, scaler_usia, le_gender, le_provinsi, ohe, X_train, y_train, extract_mfcc_func = load_and_preprocess_data()

# Load the trained model
model_path = "model_aksen.keras"

@st.cache_resource
def load_my_model(model_path):
    try:
        custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        st.success("Model successfully loaded!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

pn_model = load_my_model(model_path)

# Function to prepare audio for prediction (now uses `extract_mfcc_func` from cached data)
def process_audio_for_prediction(audio_file_path, user_usia, user_gender, user_provinsi):
    mfcc_features = extract_mfcc_func(audio_file_path) # Use the cached function
    if mfcc_features is None:
        return None

    # Scale user usia
    user_usia_scaled = scaler_usia.transform(np.array([[user_usia]]))

    # One-hot encode combined metadata (gender and provinsi)
    user_meta_for_ohe = np.array([[user_gender, user_provinsi]])
    user_cat_encoded = ohe.transform(user_meta_for_ohe)

    # Combine all metadata features
    user_X_meta = np.hstack([user_usia_scaled, user_cat_encoded]).astype(np.float32)

    mfcc_features = np.expand_dims(mfcc_features, axis=0) # Add batch dimension
    X_meta_broadcast = np.repeat(user_X_meta[:, np.newaxis, np.newaxis, :],
                                 mfcc_features.shape[1],
                                 axis=1)
    X_meta_broadcast = np.repeat(X_meta_broadcast,
                                 mfcc_features.shape[2],
                                 axis=2)

    X_final_pred = np.concatenate([mfcc_features, X_meta_broadcast], axis=-1).astype(np.float32)
    return X_final_pred

# Pre-compute prototypes for each accent class from training data
if pn_model is not None and X_train is not None and y_train is not None and le_y is not None:
    st.write("Pre-computing class prototypes...")
    train_embeddings = pn_model.embedding(X_train)
    class_prototypes = []
    class_names = le_y.classes_
    for i in range(len(class_names)):
        mask = (y_train == i)
        if np.any(mask):
            class_embeddings = train_embeddings[mask]
            prototype = tf.reduce_mean(class_embeddings, axis=0)
            class_prototypes.append(prototype)
        else:
            st.warning(f"No training samples found for class {class_names[i]}")
            class_prototypes.append(tf.zeros(train_embeddings.shape[-1])) # Placeholder
    class_prototypes = tf.stack(class_prototypes)
    st.success("Class prototypes computed.")
else:
    st.error("Model (pn_model) or training data (X_train, y_train, le_y) not available. Cannot compute prototypes.")
    st.stop()


# Streamlit UI
st.set_page_config(layout="wide")
st.title("Voice Accent Classification")
st.write("Upload an audio file and provide metadata to predict the accent.")

# Metadata Inputs
st.sidebar.header("User Metadata")
user_usia = st.sidebar.slider("Usia (Age)", min_value=10, max_value=80, value=30)
user_gender = st.sidebar.selectbox("Jenis Kelamin (Gender)", options=le_gender.classes_.tolist())
user_provinsi = st.sidebar.selectbox("Provinsi Asal (Origin Province)", options=le_provinsi.classes_.tolist())

uploaded_file = st.file_uploader("Choose a WAV audio file", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        audio_file_path = tmp_file.name

    st.audio(audio_file_path, format='audio/wav')

    st.subheader("Processing Audio and Metadata...")
    processed_audio_input = process_audio_for_prediction(audio_file_path, user_usia, user_gender, user_provinsi)

    if processed_audio_input is not None:
        # Display Spectrogram
        st.subheader("Mel-frequency Spectrogram")
        y_uploaded, sr_uploaded = librosa.load(audio_file_path, sr=22050)
        S = librosa.feature.melspectrogram(y=y_uploaded, sr=sr_uploaded)
        S_dB = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr_uploaded, x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(format='%+2.0f dB', ax=ax)
        ax.set_title('Mel-frequency spectrogram')
        st.pyplot(fig)
        plt.close(fig) # Close figure to prevent display issues

        st.subheader("Prediction:")
        if pn_model is not None and class_prototypes is not None:
            # Get embedding for the uploaded audio
            query_embedding = pn_model.embedding(processed_audio_input)

            # Calculate Euclidean distances to pre-computed prototypes
            distances = tf.norm(class_prototypes - query_embedding, axis=1)

            # Predict the class with the minimum distance
            predicted_class_idx = tf.argmin(distances).numpy()
            predicted_accent = le_y.inverse_transform([predicted_class_idx])[0]

            st.success(f"Predicted Accent (label_aksen): **{predicted_accent}**")
            st.info(f"**Note:** The current model is primarily trained for accent classification. Prediction for 'Usia', 'Gender', and 'Provinsi' would require a dedicated multi-task learning model and retraining. The metadata provided is incorporated into the input features, but its direct impact on specific predictions for 'Usia', 'Gender', and 'Provinsi' is not explicitly modeled in this prototypical network.")

        else:
            st.warning("Model or prototypes not available for prediction.")
    else:
        st.error("Could not process the audio file.")

    # Clean up the temporary file
    os.remove(audio_file_path)
