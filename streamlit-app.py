import os
print(os.getcwd())  # Cek direktori kerja saat ini
print(os.listdir())  # Lihat semua file di direktori

@st.cache_resource
def load_accent_model():
    import tensorflow as tf
    
    # Coba beberapa kemungkinan path
    possible_paths = [
        "model_aksen.keras",
        "./model_aksen.keras",
        os.path.join(os.getcwd(), "model_aksen.keras"),
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        st.sidebar.error("❌ File 'model_aksen.keras' tidak ditemukan")
        st.sidebar.info(f"📂 Direktori saat ini: {os.getcwd()}")
        st.sidebar.info(f"📄 File tersedia: {os.listdir()}")
        return None
    
    # Register custom objects
    custom_objects = register_custom_objects()
    
    try:
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects, 
            compile=False
        )
        st.sidebar.success(f"✅ Model loaded dari {model_path}")
        return model
    except Exception as e:
        try:
            model = tf.keras.models.load_model(
                model_path, 
                custom_objects=custom_objects, 
                compile=False, 
                safe_mode=False
            )
            st.sidebar.success(f"✅ Model loaded (safe_mode=False)")
            return model
        except Exception as e2:
            st.sidebar.error(f"❌ Gagal load model:")
            st.sidebar.code(str(e2))
            return None

# Tambahkan di bagian sidebar
with st.sidebar:
    st.subheader("⚙️ Settings")
    
    if model is None:
        st.warning("Model tidak terdeteksi")
        uploaded_model = st.file_uploader(
            "Upload model (.keras)", 
            type=["keras"]
        )
        
        if uploaded_model:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as f:
                f.write(uploaded_model.getbuffer())
                temp_model_path = f.name
            
            try:
                model = tf.keras.models.load_model(
                    temp_model_path,
                    custom_objects=register_custom_objects(),
                    compile=False
                )
                st.success("✅ Model uploaded & loaded!")
            except Exception as e:
                st.error(f"Error: {e}")
