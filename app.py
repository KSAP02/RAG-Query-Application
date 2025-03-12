import os
import fitz  # PyMuPDF for PDF extraction
from dotenv import load_dotenv
import requests
import faiss
import numpy as np
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.docstore.document import Document as LangchainDocument
from langchain.vectorstores.faiss import index_to_docstore_id
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

load_dotenv() # load the env content

api_key = os.getenv('HF_API_KEY')
os.environ['HF_API_KEY'] = api_key

# === CONFIGURATION ===
HF_API_KEY = api_key  # Set your Hugging Face API key here
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/mt5-base"  # You can change this to a supported HF inference model
PDF_DIR = "data"  # Directory containing PDF files

def load_pdfs_from_directory(pdf_dir):
    documents = []
    for file_name in os.listdir(pdf_dir):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(pdf_dir, file_name)
            with fitz.open(file_path) as doc:
                text = "\n".join([page.get_text() for page in doc])
                documents.append(Document(page_content=text, metadata={"source":file_name}))
    return documents

def get_hf_embeddings(texts):
    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"
    headers = {
        'Authorization': f'Bearer {HF_API_KEY}'
        }
    response = requests.post(url, headers=headers, json={"inputs":texts})
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching embeddings: {response.json()}")
    
def create_faiss_vectorstore(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    texts = [chunk.page_content for chunk in chunks]
    embeddings = get_hf_embeddings(texts)
    
    dimension = len(embeddings[0]) # Get embeddings size
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(embeddings, dtype=np.float32))
    
    # Create docstore and index_to_docstore_id
    docstore = InMemoryDocstore({i: LangchainDocument(page_content=chunk.page_content) for i, chunk in enumerate(chunks)})
    index_to_docstore_id_map = index_to_docstore_id.IndexToDocstoreID({i: i for i in range(len(chunks))})
    
    return FAISS(index=faiss_index, embedding_function=lambda x: get_hf_embeddings(x)), chunks

def retrieve_context(query, vector_store, chunks):
    query_embedding = get_hf_embeddings([query])[0]
    distances, indices = vector_store.index.search(np.array([query_embedding], dtype=np.float32), k=3)
    
    retrieved_texts = [chunks[i].page_content for i in indices[0] if i < len(chunks)]
    return "\n\n".join(retrieved_texts)

def query_hf_llm(prompt):
    url = f"https://api-inference.huggingface.co/models/{LLM_MODEL}"
    headers = {
        'Authorization': f'Bearer {HF_API_KEY}'
        }
    payload = {"inputs":prompt}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        raise Exception(f"Error fetching response: {response.json()}")
    
def main():
    print("Loading PDFs...")
    documents = load_pdfs_from_directory(PDF_DIR)
    
    print("Creating FAISS vector database...")
    vector_store, chunks = create_faiss_vectorstore(documents)

    while True:
        user_query = input("Enter a query: ")
        if user_query == "exit":
            break
        
        print("Retrieveing similar documents...")
        context = retrieve_context(user_query, vector_store, chunks)
        
        prompt = f"""Answer the following based on the provided context:
        \n\n Context:\n{context}
        \n\n Question: {user_query}"""
        
        print("Generating Response...")
        response = query_hf_llm(prompt)
        
        print(f"\n Answer: {response}")
        
# RUN APP
if __name__ == "__main__":
    main()
           