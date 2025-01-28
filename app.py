import streamlit as st
from PIL import Image
import numpy as np
from tensorflow import keras

# Set up the page
st.set_page_config(
    page_title="Alzheimer's Disease Classification",
    page_icon="🧠",
    layout="centered"
)

# App title
st.title("🧠 Alzheimer's Disease Classification")
st.markdown("""
    ### 👋 Welcome!
    Upload an image to classify the stage of Alzheimer's disease. The model predicts the level of impairment based on the uploaded image.
""")

# Cache the model loading process
@st.cache_resource
def load_model():
    return keras.models.load_model("model.keras")

model = load_model()

# Function to preprocess the image
def preprocess_image(image_file):
    image = Image.open(image_file)
    image = image.resize((128, 128))
    image_arr = np.array(image.convert("RGB"))
    image_arr = image_arr / 255.0  # Normalize the image
    image_arr = np.expand_dims(image_arr, axis=0)
    return image_arr

# Sidebar for image upload
st.sidebar.title("📤 Upload Image")
image_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

# Sidebar instructions
st.sidebar.markdown("""
    **Instructions:**
    - Make sure the image is clear and well-lit.
    - Supported formats: JPG, PNG, JPEG.
""")

if image_file is not None:
    # Display the uploaded image
    st.image(image_file, caption="📸 Uploaded Image", use_column_width=True)

    # Preprocess the image
    image_arr = preprocess_image(image_file)

    # Make a prediction
    with st.spinner("🔍 Analyzing the image..."):
        result = model.predict(image_arr)
        ind = np.argmax(result)

    # Define the classes
    classes = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

    # Display the prediction
    st.success(f"✅ **Prediction:** {classes[ind]}")

    # User tip
    st.info("💡 For the best results, ensure the image is clear and of high quality.")

else:
    st.warning("⚠️ Please upload an image for classification.")

# Footer design
st.markdown("""
---
🎯 **Developed using [Streamlit](https://streamlit.io) and TensorFlow.**
""")
