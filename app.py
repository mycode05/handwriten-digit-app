import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

# Load the trained model
model = load_model('mnist.h5')

# Set page configuration
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main content
st.title('MNIST Digit Classifier')
st.write(
    """
    Upload an image of a handwritten digit (0-9), and this app will classify it using a Convolutional Neural Network (CNN) trained on the MNIST dataset.
    """
)

# Function to preprocess the image
def preprocess_image(image):
    # Convert the image to grayscale
    image = ImageOps.grayscale(image)
    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))
    # Convert the image to a numpy array
    image = np.array(image)
    # Invert the image colors (MNIST dataset has white digits on a black background)
    image = 255 - image
    # Normalize the image
    image = image.astype('float32') / 255.0
    # Reshape the image to match the input shape required by the model
    image = image.reshape(1, 28, 28, 1)
    return image

# Footer
st.write(
    """
    Upload Image
    Choose an image file

    Drag and drop file here
    Limit 200MB per file • PNG, JPG, JPEG
    """
)

# File uploader at the bottom of the page
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

# Display classification results
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess and classify image
    image = Image.open(uploaded_file)
    processed_image = preprocess_image(image)

    # Make a prediction
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)

    st.write(f'The digit is classified as: **{predicted_digit}**')

# Footer in the sidebar
st.sidebar.markdown("---")
st.sidebar.write("**MNIST Digit Classifier App**")
st.sidebar.write("Created with Streamlit and TensorFlow.")
