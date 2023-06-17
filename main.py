'''
# ex: 1
import streamlit as st
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader

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


'''
'''

from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask Your PDF", page_icon="🗂️")
    st.title("Ask Your PDF 💬")

    # Upload the file
    pdf = st.file_uploader("Upload your PDF", type=["pdf"])

    # Extract text from the uploaded file
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    
        # Create chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Create the embeddings
        embeddings = OpenAIEmbeddings()

        # Build the vector store
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Chat functionality
        user_input = st.text_input("Ask a question about your PDF:")
        if user_input:
            docs = knowledge_base.similarity_search(user_input)

            chain = load_qa_chain(OpenAI(), chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_input)
                print(cb)

            st.write(response)


if __name__ == '__main__':
    main()

'''

import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

# Sidebar contents
with st.sidebar:
    st.title('Chat with Your Insurance Plan')
    st.markdown('''
    ## About
    Chat with your insurance plan in pdf to discover the right information for your decision make

    ''')
    add_vertical_space(5)

load_dotenv()

def main():
    st.header("Chat with pdf")

    # upload a pdf file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    
    #st.write(pdf)
    if pdf is not None:
        #pdf reader
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)
 
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
 
        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)
 
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
 
if __name__ == '__main__':
    main()

