
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("best_mobilenetv2.keras")

st.title("Fruit Freshness Detection")

st.write("Upload gambar buah untuk mengetahui apakah buah Fresh atau Stale.")

uploaded_file = st.file_uploader(
    "Upload gambar buah",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(img, caption="Gambar yang diupload")

    img = img.resize((224, 224))

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)[0][0]

    if prediction >= 0.5:
        result = "Stale"
        confidence = prediction * 100
    else:
        result = "Fresh"
        confidence = (1 - prediction) * 100

    st.subheader("Hasil Prediksi")
    st.write("Prediction:", result)
    st.write("Confidence:", round(confidence, 2), "%")
