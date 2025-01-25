import streamlit as st
from PIL import Image
import numpy as np
from tensorflow import keras

st.title("Alzheimer's Disease Classification")

# Cache the model loading process
@st.cache_resource
def load_model():
    return keras.models.load_model("model.keras")

model = load_model()

# Function to preprocess the image
def preprocess_image(image_file):
    image = Image.open(image_file)
    image = image.resize((128 ,128))
    image_arr = np.array(image.convert("RGB"))
    image_arr = image_arr / 255.0  # Normalize the image
    image_arr = np.expand_dims(image_arr, axis=0)
    return image_arr

# Upload image
image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if image_file is not None:
    st.image(image_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image_arr = preprocess_image(image_file)

    # Make a prediction
    result = model.predict(image_arr)
    ind = np.argmax(result)

    # Define classes
    classes = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

    # Display the prediction
    st.header(f"Prediction: {classes[ind]}")
