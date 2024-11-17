from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_community.vectorstores import Chroma
import google.generativeai as genai
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


def get_pages(pdf_path="data/modern OS TANENBAUM.pdf"):
    pdf_loader = PyPDFLoader(pdf_path)
    pages = pdf_loader.load_and_split()
    return pages

def split_pages(pages, batch_size=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = []
    already_split = 0
    while True:
        if len(pages) < batch_size:
            chunks.extend(text_splitter.split_text("\n".join([str(p.page_content) for p in pages])))
            break
        if already_split + batch_size > len(pages):
            chunks.extend(text_splitter.split_text("\n".join([str(p.page_content) for p in pages[already_split:]])))
            break
        else:
            current_pages = "\n".join([str(p.page_content) for p in pages[already_split:already_split+batch_size]])
            chunks.extend(text_splitter.split_text(current_pages))
        already_split += batch_size
    return chunks
   
def batch_texts(texts, batch_size):
    for i in range(0, len(texts), batch_size):
        yield texts[i:i + batch_size]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def add_to_vector_store(batch, vector_store=None):
    vector_store.add_texts(batch)
    return vector_store


def create_vector_store(GOOGLE_API_KEY):
    return Chroma(
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY),
        persist_directory="docs_vec",
    )   

def make_vector_store(GOOGLE_API_KEY, texts, batch_size=20, vector_store=None):
    if vector_store is None:
        vector_store = create_vector_store(GOOGLE_API_KEY)
    for batch in batch_texts(texts, batch_size):
        vector_store = add_to_vector_store(batch, vector_store)
    return vector_store



def get_vector_store(GOOGLE_API_KEY, pdf_path="data/modern OS TANENBAUM.pdf"):
    pages = get_pages()
    text_pages = split_pages(pages)
    vector_store = make_vector_store(GOOGLE_API_KEY, text_pages)
    print("Vector store created")
    return vector_store

if __name__ == "__main__":
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    vector_store = get_vector_store(GOOGLE_API_KEY)
    print("Vector store created")
    vector_index = vector_store.as_retriever(search_kwargs={"k": 5})




    