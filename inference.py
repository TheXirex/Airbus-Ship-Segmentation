import cv2
import random
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

from config import *
from utils.losses import *


def gen_pred(img, model):
    img = cv2.resize(img, (768, 768))
    img = img[::IMG_SCALING[0], ::IMG_SCALING[1]]
    img = img / 255
    img = tf.expand_dims(img, axis=0)
    pred = model.predict(img)
    pred = np.squeeze(pred, axis=0)
    return cv2.resize(pred, (768, 768))

def segmentation(img, pred, alpha=0.5):
    segmented_img = np.copy(img)
    threshold = 0.3

    labels = cv2.connectedComponentsWithStats(np.uint8(pred > threshold))[1]

    for label in range(1, labels.max() + 1):
        mask = labels == label
        if label < len(COLORS) + 1:
            color = COLORS[label - 1]
            segmented_img[mask] = color

    img_array = np.array(img)
    segmented_img = cv2.addWeighted(img_array, 1 - alpha, segmented_img, alpha, 0)

    return segmented_img

def main():
    st.set_page_config(layout="wide")
    st.write("<center><h2>Airbus Ship Detection</h2>", unsafe_allow_html=True)
    
    file = st.file_uploader("Upload an image for detection (image will be stretched to 768x768):", type=['jpg', 'png'])

    if file:
        image = Image.open(file)
        image_array = np.array(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        with tf.keras.utils.custom_object_scope({'focal_loss': focal_loss, 'dice_score': dice_score}):
            trained_model = tf.keras.models.load_model('weights_and_models/best.h5')

        predict = gen_pred(image_array, trained_model)

        col1, col2, col3 = st.columns([1,1,1])
        
        with col1:
            st.markdown("<center><h5>Original Image</h5>", unsafe_allow_html=True)
            st.image(image)

        with col2:
            st.markdown("<center><h5>Predicted Image</h5>", unsafe_allow_html=True)
            st.image(cv2.cvtColor(predict, cv2.COLOR_RGB2BGR))
        
        with col3:
            st.markdown("<center><h5>Segmented Image</h5>", unsafe_allow_html=True)
            segmented_img = segmentation(image, predict)
            st.image(segmented_img)
    else:
        st.text('No image has been uploaded.')

if __name__ == '__main__':
    main()