import streamlit as st 
from PIL import Image
import tensorflow as tf 

print(tf.__version__)

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load the trained model
model_mnist = tf.keras.models.load_model('mnist_new.h5')

print(model_mnist.summary())

# The User Interface for Streamlit
st.title("MNIST Digit Classifier")
st.write("Draw a digit in the box below and get the prediction!")

# Canvas for user input
canvas = st.file_uploader("Upload a 28x28 grayscale image of a digit", type=["png", "jpg", "jpeg"])

# reformatting is essential to avoid any error
if canvas is not None:
    image = Image.open(canvas).convert('L')  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert (black bg to white bg)
    image = image.resize((28, 28))  # Resize to match MNIST format
    image_array = np.array(image) / 255.0  # Normalize
    image_array = image_array.reshape(1, 28, 28)  # Reshape for model

       # Predict
    prediction = model_mnist.predict(image_array)
    predicted_digit = np.argmax(prediction)

    # Show the image and result
    st.image(image, caption=f"Predicted Digit: {predicted_digit}", width=150)