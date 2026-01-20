import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from class_names import class_names

# Page title
st.set_page_config(page_title="Plant Disease Analysis", page_icon="🌿")

st.title("🌿 Plant Disease Analysis System")
st.write("Upload a leaf image to detect plant disease")

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.h5")

model = load_model()

# Class names
# class_names = sorted(os.listdir("PlantVillage/train"))

# Treatment dictionary
treatment_dict = {
    "Pepper__bell___Bacterial_spot":
        "Use copper-based fungicides. Avoid overhead irrigation. Remove infected leaves.",

    "Pepper__bell___healthy":
        "Plant is healthy. Maintain proper watering and nutrients.",

    "Tomato___Late_blight":
        "Apply fungicides like chlorothalonil or mancozeb. Remove infected leaves.",

    "Tomato___Early_blight":
        "Use fungicide sprays and practice crop rotation.",

    "Tomato___healthy":
        "Plant is healthy. Ensure proper sunlight and soil care."
}

# Upload image
uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    if st.button("🔍 Predict Disease"):
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        index = np.argmax(predictions)
        confidence = predictions[0][index]

        disease = class_names[index]
        treatment = treatment_dict.get(
            disease,
            "No treatment information available. Consult an agricultural expert."
        )

        st.success(f"🌱 Disease Detected: {disease}")
        st.info(f"✅ Confidence: {confidence * 100:.2f}%")
        st.warning(f"💊 Treatment: {treatment}")
