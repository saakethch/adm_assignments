import os
import streamlit as st
import openai
import config

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
    tshirt_type = st.text_input("Enter the custome style for T-shirt:", "")
    image_type = tshirt_type

    if image_type!="":
        generate_image(image_type)