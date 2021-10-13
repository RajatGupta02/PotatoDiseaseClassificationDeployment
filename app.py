import streamlit as st
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def predict(model, img):
    class_names={
        0 : "Early Blight",
        1: "Late Blight",
        2: "Healthy"
    }
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis = 0) #Since the input should be of the form of (batch_size, imagesize, imagesize, 3)
    
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    
    confidence = round(100*(np.max(predictions[0])) ,2)
    
    return predicted_class, confidence


# MODEL = tf.keras.models.load_model('saved_models/1')
MODEL = tf.keras.models.load_model('webmodel.h5')

st.title("Potato Disease Classification")
st.subheader('Upload an Image of Potato Leaf')

uploaded_image= st.file_uploader("Choose an image...", type=['jpg', 'png'])

if uploaded_image is not None:
    uploaded_image = Image.open(uploaded_image)
    st.image(uploaded_image)
    result, confidence=predict(MODEL, uploaded_image)

    st.markdown(f'**Prediction**: {result}')
    st.markdown(f'**Confidence**: {confidence}%')
    