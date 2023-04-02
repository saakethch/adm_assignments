import streamlit as st
from task1 import task1
from task2 import task2
from task3 import task3



# Define the Streamlit app
def main():

    st.title("Assignment 4 - Image Search")
    st.subheader("Saaketh Chaganty, Disha Vaswani")

    tab1, tab2, tab3 = st.tabs(["Image search", "OpenAI", "Pinecone"])

    with tab1:
        task1()
    with tab2:
        task2()
    with tab3:
        task3()

if __name__ == "__main__":
    main()