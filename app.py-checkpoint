
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import os

# Set the title of the Streamlit app
st.title('Brain Tumor Detection App')

# Load the trained model
@st.cache_resource
def load_my_model():
    model = load_model('model.h5')
    return model

model = load_my_model()

# Define class labels (ensure these match your training labels)
LABELS = ['glioma', 'meningioma', 'no_tumor', 'pituitary'] # Make sure this matches your LABELS variable from the notebook
IMAGE_SIZE = 128 # Make sure this matches your IMAGE_SIZE variable from the notebook

def preprocess_image(image):
    # Resize image to target size
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array / 255.0 # Normalize pixel values
    return img_array

def predict_image(image_array):
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]
    return predicted_class_index, confidence_score

# File uploader widget
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess and predict
    processed_image = preprocess_image(image)
    predicted_class_index, confidence_score = predict_image(processed_image)

    # Display prediction results
    predicted_label = LABELS[predicted_class_index]
    
    if predicted_label == 'no_tumor':
        st.success(f"Prediction: No Tumor (Confidence: {confidence_score * 100:.2f}%) 🥳")
    else:
        st.error(f"Prediction: Tumor - {predicted_label.replace('_', ' ').title()} (Confidence: {confidence_score * 100:.2f}%) 🚨")
