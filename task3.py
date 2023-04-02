import numpy as np
from PIL import Image
from datetime import datetime
from pathlib import Path
import streamlit as st
import pinecone
import torch
import config
import torchvision
from torchvision.transforms import (
    Compose, 
    Resize, 
    CenterCrop, 
    ToTensor, 
    Normalize
)

def task3():
    st.title("Image Search 10k")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"],key="task_3")
    preprocess = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if uploaded_file is not None:
        start_time = datetime.now()
        img = Image.open(uploaded_file)
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", "_") + uploaded_file.name
        img.save(uploaded_img_path)
        pc_api_key = config.api_keys["PINECONE_API_KEY"]
        pinecone.init(api_key=pc_api_key, environment="eu-west1-gcp")
        model = torchvision.models.squeezenet1_1(pretrained=True).eval()
        index = pinecone.Index("pinecone-image-search")

        query_vectors = model(preprocess(img).unsqueeze(0)).tolist()
        responses = index.query(vector=query_vectors, top_k=3)
        end_time = datetime.now()
        st.write('Search time: {}'.format(end_time - start_time))
        
        for res in responses["matches"]:
            images = Image.open(f'static/img_data/{res["id"]}')
            score = res["score"]
            print(score)
            print("here", res["id"])
            st.image(images, caption=f"L2 Distance: {score:.2f}", use_column_width="auto")