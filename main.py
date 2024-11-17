import google.generativeai as genai
import os

from langchain_google_genai import ChatGoogleGenerativeAI

import urllib
import warnings

import pandas as pd
import langchain_chroma

from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from dotenv import find_dotenv, load_dotenv
from src.load_data import get_vector_store

load_dotenv(find_dotenv())
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
print(GOOGLE_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)
model_name = "gemini-1.5-flash"

# Model initialization
model = ChatGoogleGenerativeAI(model=model_name,google_api_key=GOOGLE_API_KEY,
                             temperature=0.1, max_retries=2)

# Vector store initialization
vector_store = get_vector_store(GOOGLE_API_KEY)
vector_index = vector_store.as_retriever(search_kwargs={"k": 5})


template = """You are a Professor of Operating Systems. You can use Modern Operating Systems by Tanenbaum Book to answer questions about Operating Systems.
Use the following pieces of context in the book to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain
qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vector_index,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)


input_question = input("Enter your question or 0 to exit: ")




while input_question != "0":
    result = qa_chain({"query": input_question})
    print(result["result"])
    input_question = input("Enter your question or 0 to exit: ")




