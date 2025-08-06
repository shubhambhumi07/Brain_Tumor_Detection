import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 📌 Constants
IMG_SIZE = 300
THRESHOLD = 0.6  # More confident threshold

# 📌 Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_models/brain_tumor_model.h5", compile=False)

model = load_model()

# 📌 Streamlit Setup
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image to detect if it contains signs of a brain tumor.")

# 📤 Upload image
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 🖼️ Show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)  # ✅ use_container_width instead of deprecated use_column_width

    # 🌀 Preprocess image
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0  # ✅ Normalized as in training
    img_batch = np.expand_dims(img_array, axis=0)

    # 🔍 Predict
    st.write("🔍 Predicting...")
    prediction = model.predict(img_batch)[0][0]

    # 🧠 Display Result
    label = "🚨 Brain Tumor Detected" if prediction >= THRESHOLD else "✅ No Brain Tumor Detected"
    confidence = prediction if prediction >= THRESHOLD else 1 - prediction

    st.subheader(label)
    st.write(f"🧪 Confidence: **{confidence * 100:.2f}%**")
   


