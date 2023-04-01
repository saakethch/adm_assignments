import streamlit as st
from task1 import task1
from task2 import task2
from task3 import task3



# Define the Streamlit app
def main():

    st.title("Assignment 4")

    tab1, tab2, tab3 = st.tabs(["Task 1", "Task 2", "Task 3"])

    with tab1:
        task1()
    with tab2:
        task2()
    with tab3:
        task3()

if __name__ == "__main__":
    main()