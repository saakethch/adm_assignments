import os
import streamlit as st
import openai
import config
import time 
openai.api_key = config.api_keys["OPENAI_API_KEY"]
def generate_image(image_type):
    response = openai.Image.create(
    prompt= image_type,
    n=1,
    size="1024x1024"
    )
    image1 = response['data'][0]['url']
    st.image(image=image1, caption="Your customized fashion T-Shirt")

def task2():
    tshirt_type = st.text_input("Enter the custom style for T-shirt:", "") + "printed on a"+ st.selectbox(f'Apparel type', options=['Blazer','Suit','Shirt', 'T-Shirt','Hoodie','Shorts','Jeans'])
    image_type = tshirt_type
    time.sleep(3)
    if image_type!="":
        generate_image(image_type)