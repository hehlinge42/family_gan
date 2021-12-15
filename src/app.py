from pathlib import WindowsPath
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image 

import tensorflow_hub as hub
import tensorflow as tf
from tensorflow import keras

@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img 

# @st.cache
def load_model_son(model_path):
    model_son = tf.keras.models.load_model(model_path,  compile=False)
    return model_son

# @st.cache
def load_model_daughter(model_path):
    model_daughter = tf.keras.models.load_model(model_path,  compile=False)
    return model_daughter

st.title('Gan he make a good husband?')

with st.container():
    st.header("Please choose a picture of the wonderful parents...")
    col1, col2 = st.columns(2)
    with col1: 
        uploaded_file_one = st.file_uploader("Father")
        if uploaded_file_one is not None:
            parent_one = uploaded_file_one.getvalue() # To read file as bytes:
            parent_one_img = load_image(uploaded_file_one)
            st.image(parent_one_img, width=250)

    with col2:
        uploaded_file_two = st.file_uploader("Mother")
        if uploaded_file_two is not None:
            parent_two = uploaded_file_two.getvalue() # To read file as bytes:
            parent_two_img = load_image(uploaded_file_two)
            st.image(parent_two_img, width=250)


with st.container():
    st.header("... and their gorgeous children look like:")
    model_son = load_model_son("./data/models/pix2pix_son_v1")
    # model_daughter = load_model_daughter("../data/models/pix2pix_daughter_v1")
    if st.button('Generate Children'):
        col1, col2 = st.columns(2)
        with col1:
            st.text("son")
            parent_one_tensor = tf.io.decode_image(parent_one)
            parent_two_tensor = tf.io.decode_image(parent_two)
            parent_one_tensor_expanded = tf.expand_dims(parent_one_tensor, axis=0)
            parent_two_tensor_expanded = tf.expand_dims(parent_two_tensor, axis=0)
            son = model_son((parent_one_tensor_expanded, parent_two_tensor_expanded))
            st.image(son, width=250)

        with col2:
            st.text("daughter")
            # daughter = model_son(parent_one_img, parent_two_img)
            # st.image(daughter, width=250)
    
    