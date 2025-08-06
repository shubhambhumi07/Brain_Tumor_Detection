import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 📌 Constants
IMG_SIZE = 224  # ✅ Must match training input size
THRESHOLD = 0.6  # Customize if needed

# 📌 Load model (cached)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("trained_models/brain_tumor_model.h5", compile=False)
    return model

model = load_model()

# 📌 Streamlit Setup
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image to check for signs of a brain tumor.")

# 📤 Upload image
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 🖼️ Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    # 🌀 Preprocess image (MUST match training preprocessing)
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0  # ✅ MATCHES TRAINING NORMALIZATION
    img_batch = np.expand_dims(img_array, axis=0)

    # 🔍 Predict
    st.write("🔍 Predicting...")
    prediction = model.predict(img_batch)[0][0]  # Binary classification output

    # 🧠 Show result
    if prediction >= THRESHOLD:
        label = "🚨 Brain Tumor Detected"
        confidence = prediction
    else:
        label = "✅ No Brain Tumor Detected"
        confidence = 1 - prediction

    st.subheader(label)
    st.write(f"🧪 Confidence: **{confidence * 100:.2f}%**")