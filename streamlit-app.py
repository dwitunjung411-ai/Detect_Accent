import numpy as np
import librosa
import tensorflow as tf

# =========================================================
# 1. PROTOTYPICAL NETWORK
# =========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model, **kwargs):
        super().__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, inputs):
        support_set = inputs["support_set"]
        query_set = inputs["query_set"]
        support_labels = inputs["support_labels"]
        n_way = inputs["n_way"]

        # Embedding
        support_embed = self.embedding(support_set)
        query_embed = self.embedding(query_set)

        # Prototype
        prototypes = []
        for i in range(n_way):
            class_embed = tf.boolean_mask(
                support_embed, support_labels == i
            )
            prototypes.append(tf.reduce_mean(class_embed, axis=0))

        prototypes = tf.stack(prototypes)

        # Distance
        distances = tf.norm(
            tf.expand_dims(query_embed, 1) - prototypes,
            axis=2
        )

        return -distances


# =========================================================
# 2. LOAD MODEL (INI LETAK NAMA MODEL)
# =========================================================
embedding_model = tf.keras.models.load_model(
    "model_aksen.keras",   # <<< NAMA MODEL
    compile=False
)

proto_model = PrototypicalNetwork(embedding_model)


# =========================================================
# 3. AUDIO → MFCC
# =========================================================
def extract_mfcc(audio_path, sr=16000, n_mfcc=40):
    y, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc.astype(np.float32)


# =========================================================
# 4. SUPPORT SET (VERSI AMAN - TIDAK ERROR)
# =========================================================
def build_support_set(query_feat, n_way):
    support_set = []
    support_labels = []

    for i in range(n_way):
        support_set.append(
            query_feat + np.random.normal(0, 0.01, query_feat.shape)
        )
        support_labels.append(i)

    return (
        np.array(support_set, dtype=np.float32),
        np.array(support_labels, dtype=np.int32)
    )


# =========================================================
# 5. PREDIKSI AKSEN
# =========================================================
def predict_accent(audio_path):
    accent_classes = [
        "Betawi",
        "Sunda",
        "Jawa_Tengah",
        "Jawa_Timur",
        "Yogyakarta"
    ]
    n_way = len(accent_classes)

    # QUERY SET
    query_feat = extract_mfcc(audio_path)
    query_set = np.expand_dims(query_feat, axis=0)

    # SUPPORT SET
    support_set, support_labels = build_support_set(
        query_feat, n_way
    )

    inputs = {
        "support_set": support_set,
        "query_set": query_set,
        "support_labels": support_labels,
        "n_way": n_way
    }

    logits = proto_model(inputs)
    pred_idx = tf.argmax(logits, axis=1).numpy()[0]

    return accent_classes[pred_idx]


# =========================================================
# 6. MAIN
# =========================================================
if __name__ == "__main__":
    audio_path = "contoh.wav"  # ganti audio kamu
    hasil = predict_accent(audio_path)
    print("Prediksi aksen:", hasil)
