import pinecone
import os
import requests
import shortuuid
import tqdm
import numpy as np
from PIL import Image
import config
import torch
import torchvision
from torchvision.transforms import (
    Compose, 
    Resize, 
    CenterCrop, 
    ToTensor, 
    Normalize
)

def process_images(img_dir, model):
    vectors = []
    preprocess = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.543, 0.546, 0.451], std=[0.312, 0.265, 0.257]),
    ])
    ctr = 0
    for file_name in os.listdir(img_dir):
        if file_name.lower().endswith(('.jpg', '.jpeg')) and ctr<990:
            ctr+=1
            print(ctr)
            img_path = os.path.join(img_dir, file_name)
            img = Image.open(img_path)
            embedding = model(preprocess(img).unsqueeze(0)).tolist()
            vector = {'id':str(file_name), 'values':embedding[0]}
            vectors.append(vector)
        else:
            break
    return vectors

def create_index(index_name,dim):
    pinecone_api_key = config.api_keys["PINECONE_API_KEY"]
    pinecone.init(api_key=pinecone_api_key, environment="eu-west1-gcp")
    model = torchvision.models.squeezenet1_1(pretrained=True).eval()
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, dimension=dim)
    index = pinecone.Index(index_name)
    vectors = process_images(img_dir="images/img_data",model=model)
    batches = []
    for i in range(0, len(vectors), 50):
        batch = vectors[i:i+50]
        batches.append(batch)
    # Batch upsert
    for b in batches:
        print("batch upsert")
        index.upsert(b)

    pc_res  = index.describe_index_stats()
    return pc_res

INDEX_NAME = 'pinecone-image-search'
INDEX_DIM = 1000

print(create_index(INDEX_NAME,INDEX_DIM))