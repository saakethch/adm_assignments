import streamlit as st
import cv2
from PIL import Image
import requests
from io import BytesIO
import os
from pathlib import Path
from feature_extractor import FeatureExtractor
import numpy as np
from datetime import datetime

# Define the path to the image directory
IMG_DIR = "static"

# Define the Streamlit app
def task1():
    st.title("Visual Search Application")
    st.write("Upload an image and we'll find similar images for you!")
    
    # Create a file uploader in Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Load features 
    fe = FeatureExtractor()
    features = []
    img_paths = []
    for feature_path in Path("./static/feature").glob("*.npy"):
        features.append(np.load(feature_path))
        img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
    features = np.array(features)

    if uploaded_file is not None:
        start_time = datetime.now()

        # Load the image from the uploaded file
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Save the image to the images directory
        img_path = os.path.join(IMG_DIR, "uploaded/", uploaded_file.name)
        image.save(img_path)

        # Run search
        query = fe.extract(image)
        dists = np.linalg.norm(features-query, axis=1) 
        ids = np.argsort(dists)[:10]  
        scores = [(dists[id], img_paths[id]) for id in ids]
        end_time = datetime.now()
        st.write('Search time: {}'.format(end_time - start_time))

        # Display most similar 10 results
        st.image(uploaded_file, caption="Query Image", use_column_width="auto")
        for score in scores:
            st.image(str(score[1]), caption=f"L2 Distance: {score[0]:.2f}", use_column_width="true")
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
