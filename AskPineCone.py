import os
import sys
import time
import openai
from typing import Any, Dict, List
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
from langchain.prompts.chat import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)
import pinecone

openai.api_type = "azure"
openai.api_base = "https://cog-2iwhormj3dgc4.openai.azure.com/"
openai.api_version = "2023-08-01-preview"
openai.api_key = "ffd6ab9f00c94c52b665b53a25f5df5f"

pinecone.init(
    api_key="9bff356b-0a2b-42ca-af90-ec0267ab3a40",
    environment="gcp-starter",
)

#INDEX_NAME = "langchain-doc-index"

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):

    embeddings = OpenAIEmbeddings(deployment="embedding", openai_api_key="ffd6ab9f00c94c52b665b53a25f5df5f")
    docsearch = Pinecone.from_existing_index(
        embedding=embeddings,
        index_name="consultaide",
    )
    chat = AzureChatOpenAI(
        temperature = 0.7,
        openai_api_key = openai.api_key,
        openai_api_base = openai.api_base,
        openai_api_version = openai.api_version,
        openai_api_type = openai.api_type,
        deployment_name = "chat")

    # Create a chain that uses the Chroma vector store
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = db.as_retriever(),
        return_source_documents = True,
        chain_type_kwargs = chain_type_kwargs
    )
    chat = AzureChatOpenAI(deployment_name="chat", 
                           openai_api_key="ffd6ab9f00c94c52b665b53a25f5df5f", 
                           openai_api_base="https://cog-2iwhormj3dgc4.openai.azure.com/", 
                           openai_api_version= "2023-08-01-preview",
                           model_name="gpt-35-turbo",temperature=0.7)

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        chain_type="stuff"
    )
    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm(query="Who are the consultants involved in these")) 