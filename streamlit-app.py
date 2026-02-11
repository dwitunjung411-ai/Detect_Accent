import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
import tensorflow as tf
import os
import tempfile

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


# Feature extraction function (copied from notebook)
def extract_mfcc(file_path, sr=22050, n_mfcc=40, max_len=174):
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

# Function to prepare audio for prediction
def process_audio_for_prediction(audio_file_path, metadata_input):
    mfcc_features = extract_mfcc(audio_file_path)
    if mfcc_features is None:
        return None
    dummy_X_meta = np.zeros((1, 8), dtype=np.float32) # Replace with actual metadata if available

    mfcc_features = np.expand_dims(mfcc_features, axis=0) # Add batch dimension
    X_meta_broadcast = np.repeat(dummy_X_meta[:, np.newaxis, np.newaxis, :],
                                 mfcc_features.shape[1],
                                 axis=1)
    X_meta_broadcast = np.repeat(X_meta_broadcast,
                                 mfcc_features.shape[2],
                                 axis=2)

    X_final_pred = np.concatenate([mfcc_features, X_meta_broadcast], axis=-1).astype(np.float32)
    return X_final_pred

# Pre-compute prototypes for each accent class from training data
# This assumes X_train and y_train are available from the notebook's execution context.
# If running as a standalone script, you'd need to load or re-create these.

if 'X_train' in locals() and 'y_train' in locals() and 'le_y' in locals() and pn_model is not None:
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
    st.error("Training data (X_train, y_train, le_y) or model not found in context. Cannot compute prototypes.")
    st.stop()


# Streamlit UI
st.set_page_config(layout="wide")
st.title("Voice Accent Classification")
st.write("Upload an audio file to predict the accent.")

uploaded_file = st.file_uploader("Choose a WAV audio file", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        audio_file_path = tmp_file.name

    st.audio(audio_file_path, format='audio/wav')

    st.subheader("Processing Audio...")
    # For the UI, we're not asking for metadata explicitly for this prediction demo.
    # The process_audio_for_prediction uses a dummy metadata_input.
    processed_audio_input = process_audio_for_prediction(audio_file_path, metadata_input=None)

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

            st.success(f"Predicted Accent: **{predicted_accent}**")
        else:
            st.warning("Model or prototypes not available for prediction.")
    else:
        st.error("Could not process the audio file.")

    # Clean up the temporary file
    os.remove(audio_file_path)
```

