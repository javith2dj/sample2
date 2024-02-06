import os
import openai

from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.llms.openai import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone
import pinecone

import streamlit as st

# Load environment variables from .env file
load_dotenv(dotenv_path=".env")

openai.api_type = "azure"
openai.api_base = "https://cog-2iwhormj3dgc4.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "ffd6ab9f00c94c52b665b53a25f5df5f"

pinecone.init(
    api_key="9bff356b-0a2b-42ca-af90-ec0267ab3a40",
    environment="gcp-starter",
)


# Streamlit Code for UI - Upload PDF(s)
st.title('Demo :microphone:')

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
if uploaded_files:
    raw_text = ''
    # Loop through each uploaded file
    for uploaded_file in uploaded_files:

    # Read the PDF
     pdf_reader = PdfReader(uploaded_file)

        # Loop through each page in the PDF
    for i, page in enumerate(pdf_reader.pages):
    
            # Extract the text from the page
            text = page.extract_text()
        
            # If there is text, add it to the raw text
            if text:
              raw_text += text
              
    # Split text into smaller chucks to index them
    text_splitter = CharacterTextSplitter(chunk_size = 1500,chunk_overlap = 0)

    texts = text_splitter.split_text(raw_text)


    # Download embeddings from OPENAI
    embeddings = OpenAIEmbeddings(deployment="embedding",chunk_size=1000) # Default model "text-embedding-ada-002"
    
    Pinecone.from_texts(texts, embeddings, index_name="consultaide")
   
    # Create a FAISS vector store with all the documents and their embeddings
    docsearch = FAISS.from_texts(texts, embeddings)

    # Load the question answering chain and stuff it with the documents
    llm = AzureOpenAI(deployment_name="davinci", model_name="text-davinci-003",temperature=0.5, max_tokens=500)
    chain = load_qa_chain(llm, chain_type="refine", verbose=True) 

    query = st.text_input("Ask a question or give an instruction")

    if query:
      # Perform a similarity search to find the 6 most similar documents "chunks of text" in the corpus of documents in the vector store
      docs = docsearch.similarity_search(query, k=6)
      
      # Run the question answering chain on the 6 most similar documents based on the user's query
      answer = chain.run(input_documents=docs, question=query)
      
      # Print the answer and display the 6 most similar "chunks of text" vectors 
      st.write(answer, docs[0:6])


