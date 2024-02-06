import os
import openai
import io
import pdfplumber

from dotenv import load_dotenv
from docx import Document
from langchain.llms.openai import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import VectorDBQA
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import LatexTextSplitter
from langchain.vectorstores import Pinecone
import pinecone
from typing import Iterable
import streamlit as st

# Load environment variables from .env file
load_dotenv(dotenv_path=".env")

openai.api_type = "azure"
openai.api_base = "https://cog-2iwhormj3dgc4.openai.azure.com/"
openai.api_version = "2023-08-01-preview"
openai.api_key = "3d21de1940a849b3bd4c97c710e35f2b"
persist_directory = 'LicensingGuide'

pinecone.init(
    api_key="9bff356b-0a2b-42ca-af90-ec0267ab3a40",
    environment="gcp-starter",
)


# Streamlit Code for UI - Upload PDF(s)
st.title('Chroma Store :Upload:')

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
if uploaded_files:
    raw_text = ''
 # Create a list to store the texts of each file
    all_texts = []
    metadatas = []

    # Process each file uplodaded by the user
    for file in uploaded_files:
        # Create an in-memory buffer from the file content
        bytes = file.getvalue()

        # Get file extension
        extension = file.name.split('.')[-1]

        # Initialize the text variable
        text = ''

        # Read the file
        reader = pdfplumber.open(file)
        for i in range(len(reader.pages)):
            text +=  reader.pages[i].extract_text()

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 512,
            chunk_overlap = 10)
        texts = text_splitter.split_text(text)

        chunkmetadata = [{"source":f"{file.name}"} for i in range(len(texts))]
        # Add the chunks and metadata to the list
        all_texts.extend(texts)
        metadatas.extend(chunkmetadata) 

    # Download embeddings from OPENAI
    embeddings = OpenAIEmbeddings(deployment="embedding",chunk_size=1) # Default model "text-embedding-ada-002"
    vectordb = Chroma.from_texts(texts=all_texts, embedding=embeddings, persist_directory=persist_directory, metadatas=metadatas)

    vectordb.persist()
    vectordb = None
    

