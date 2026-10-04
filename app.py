import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Fruit Freshness Detection",
    layout="centered"
)

@st.cache_resource
def load_fruit_model():
    return load_model("best_mobilenetv2.keras")

model = load_fruit_model()

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }

    .footer {
        text-align: center;
        color: #888;
        margin-top: 40px;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">Fruit Freshness Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Fruit Freshness And Stale Classification</div>',
    unsafe_allow_html=True
)

st.info(
    "Upload a fruit image and let the AI determine whether it is Fresh or Stale."
)

uploaded_file = st.file_uploader(
    "📷 Upload your fruit image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.markdown("### Image Preview")

    st.image(
        img,
        caption="Uploaded Fruit Image",
        use_container_width=True
    )

    img_resized = img.resize((224, 224))

    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    with st.spinner("🔍 Analyzing fruit..."):
        prediction = model.predict(
            img_array,
            verbose=0
        )[0][0]

    if prediction >= 0.5:
        result = "Stale"
        confidence = prediction * 100
    else:
        result = "Fresh"
        confidence = (1 - prediction) * 100

    st.markdown("### Prediction Result")

    if result == "Fresh":

        st.success("FRESH")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

    else:

        st.error("STALE")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

    st.progress(
        min(int(confidence), 100)
    )

    st.caption(
        "Model used: MobileNetV2"
    )

st.markdown(
    """
    <div class="footer">
    Fruit Freshness Detection • Deep Learning Project
    </div>
    """,
    unsafe_allow_html=True
)
