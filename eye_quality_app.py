import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.models import Model

# === CONFIG ===
CLASSES = {0: 'bad_eye', 1: 'good_eye'}
THRESHOLD = 0.65  # Confidence threshold (only used for good_eye)

# === Load RandomForest model ===
clf = joblib.load("random_forest_eye_model.pkl")  # Ensure this file exists

@st.cache_resource
def load_feature_extractor():
    base_model = MobileNetV3Small(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    return Model(inputs=base_model.input, outputs=base_model.output)

# Call it once; result is cached
feature_extractor = load_feature_extractor()

# === Streamlit App UI ===
st.set_page_config(page_title="Eye Quality Checker", layout="centered")
st.title("👁️ Eye Image Quality Checker")
st.write("Upload an eye image. The model will predict `good_eye` or `bad_eye`, and advise retaking if needed.")

uploaded_file = st.file_uploader("📤 Upload Eye Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Eye Image", use_container_width=True)


    # Preprocess
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (224, 224))
    img_preprocessed = preprocess_input(img_resized)
    img_input = np.expand_dims(img_preprocessed, axis=0)

    # Extract features & predict
    features = feature_extractor.predict(img_input)
    pred = clf.predict(features)[0]
    prob = clf.predict_proba(features)[0][pred]
    label = CLASSES[pred]

    # Show results
    st.markdown(f"### 🧠 Prediction: **{label.upper()}**")
    st.markdown(f"**Confidence Score:** `{prob:.2f}`")

    if label == "bad_eye":
        st.error("❌ Bad image. Please retake the photo.")
    elif label == "good_eye" and prob < THRESHOLD:
        st.warning("⚠️ Low confidence in GOOD image. Consider retaking for better accuracy.")
    else:
        st.success("✅ Image is acceptable.")
