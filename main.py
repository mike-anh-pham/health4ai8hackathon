import streamlit as st
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

st.title("Welcome to Health Universe!")
st.write("This is a sample application.")

alarm_clock = st.slider('hour', 0, 23, 17) # min: 0h, max: 23h, default: 17h
st.header(alarm_clock) # print in large text

# Inputs
## Text input (not sidebar)
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")


from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("benefit-plans.pdf")
pages = loader.load_and_split()
document = pages[0]

st.write(document)
