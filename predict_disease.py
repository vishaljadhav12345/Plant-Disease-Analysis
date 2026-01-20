import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Treatment recommendations
treatment_dict = {
    "Pepper__bell___Bacterial_spot":
        "Use copper-based fungicides. Avoid overhead irrigation. Remove infected leaves.",

    "Pepper__bell___healthy":
        "Plant is healthy. Maintain proper watering and nutrient management.",

    "Tomato___Late_blight":
        "Apply fungicides like chlorothalonil or mancozeb. Remove infected plants immediately.",

    "Tomato___Early_blight":
        "Use appropriate fungicide sprays. Practice crop rotation.",

    "Tomato___healthy":
        "Plant is healthy. Ensure proper sunlight and soil nutrition."
}


# Load trained model
model = tf.keras.models.load_model("plant_disease_model.h5")

# Get class names from training folder
class_names = sorted(os.listdir("PlantVillage/train"))

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = predictions[0][predicted_index]

    return class_names[predicted_index], confidence


# -------- GIVE IMAGE PATH HERE --------
test_image = "test_leaf.jpg"

disease, confidence = predict_disease(test_image)

print("🌱 Predicted Disease:", disease)
print("✅ Confidence:", round(confidence * 100, 2), "%")

treatment = treatment_dict.get(
    disease,
    "No treatment information available. Consult an agricultural expert."
)

print("💊 Treatment Recommendation:", treatment)