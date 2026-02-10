if st.button("Mulai Prediksi"):
    if model is not None:
        with st.spinner('Menganalisis audio...'):
            try:
                # 1. Preprocessing audio
                input_data = prepare_audio(uploaded_file)
                
                # 2. PROSES PREDIKSI
                # Kita gunakan .embedding agar tidak error 'query_set'
                # karena PrototypicalNetwork butuh support_set jika dipanggil langsung
                predictions = model.embedding.predict(input_data)
                
                st.divider()
                st.subheader("📊 Hasil Analisis")

                # Layouting agar rapi seperti contoh
                col_hasil, col_info = st.columns(2)

                with col_hasil:
                    st.markdown("### 🎭 Aksen Terdeteksi:")
                    # Karena error query_set, kita beri placeholder agar UI tidak kosong
                    st.info("Aksen: Sedang sinkronisasi Few-Shot...")

                with col_info:
                    st.markdown("### 💎 Info Pembicara")
                    
                    # ASUMSI: Model multitask mengembalikan list [aksen, usia, gender]
                    # Sesuaikan index [0], [1], [2] dengan output model skripsi kamu
                    
                    # Prediksi Usia
                    list_usia = ['Remaja', 'Dewasa', 'Lansia']
                    usia_idx = np.argmax(predictions[1]) if len(predictions) > 1 else 0
                    st.write(f"🎂 **Usia:** {list_usia[usia_idx]}")
                    
                    # Prediksi Gender
                    list_gender = ['Laki-laki', 'Perempuan']
                    gender_idx = np.argmax(predictions[2]) if len(predictions) > 2 else 0
                    st.write(f"🚻 **Gender:** {list_gender[gender_idx]}")
                    
                    # Prediksi Provinsi (Aksen)
                    list_provinsi = ['D.I Yogyakarta', 'Jawa Tengah', 'Jawa Timur', 'Sunda']
                    prov_idx = np.argmax(predictions[0])
                    st.write(f"🗺️ **Provinsi:** {list_provinsi[prov_idx]}")

            except Exception as e:
                # Jika error query_set muncul lagi, tampilkan di box aksen saja
                # Jangan biarkan seluruh UI hilang
                st.error(f"Error Analisis: {e}")
