import streamlit as st

st.title("Welcome to Health Universe!")
st.write("This is a sample application.")

alarm_clock = st.slider('hour', 0, 23, 17) # min: 0h, max: 23h, default: 17h
st.header(alarm_clock) # print in large text


from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("example_data/layout-parser-paper.pdf")
pages = loader.load_and_split()
document = pages[0]
