import os
import fitz  # PyMuPDF for PDF extraction
from dotenv import load_dotenv
import requests
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore


load_dotenv() # load the env content

api_key = os.getenv('HF_API_KEY')
os.environ['HF_API_KEY'] = api_key

# === CONFIGURATION ===
HF_API_KEY = api_key  # Set your Hugging Face API key here
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"  # Can change inference model
PDF_DIR = "data"  # Directory containing PDF files


# === LOADING THE PDF DATA FROM THE DIRECTORIES INTO A LIST OF DOCUMENT DATA===
def load_pdfs_from_directory(pdf_dir):
    documents = []
    for file_name in os.listdir(pdf_dir):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(pdf_dir, file_name)
            with fitz.open(file_path) as doc:
                text = "\n".join([page.get_text() for page in doc])
                documents.append(Document(page_content=text, metadata={"source":file_name}))
    return documents

# === GETTING THE EMBEDDINGS FROM THE HUGGING FACE API ===
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
    metadatas = [{"source": f"doc_{i}"} for i in range(len(chunks))]
    # print(texts)
    embeddings = get_hf_embeddings(texts)
    # print(len(embeddings[0]))
    dimension = len(embeddings[0]) # Get embeddings size
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(embeddings, dtype=np.float32))
    
    # Create docstore (mapping between document ID and actual content)
    documents_dict = {str(i): Document(page_content=texts[i], metadata=metadatas[i]) for i in range(len(texts))}
    docstore = InMemoryDocstore(documents_dict)

    # Create index-to-document mapping
    index_to_docstore_id = {i: str(i) for i in range(len(texts))}
    
    # Wrap FAISS index with LangChain's FAISS
    vector_store = FAISS(
        index=faiss_index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        embedding_function=lambda x: get_hf_embeddings(x)  # Query embedding function
    )
    
    return vector_store, chunks
        
def retrieve_context(query, vector_store, chunks):
    query_embedding = get_hf_embeddings([query])[0]
    distances, indices = vector_store.index.search(np.array([query_embedding], dtype=np.float32), k=3)
    
    retrieved_texts = [chunks[i].page_content for i in indices[0] if i < len(chunks)]
    return "\n\n".join(retrieved_texts)

# === GETTING THE EMBEDDINGS FROM THE HUGGING FACE API ===
def query_hf_llm(prompt):
    url = f"https://api-inference.huggingface.co/models/{LLM_MODEL}"
    headers = {
        'Authorization': f'Bearer {HF_API_KEY}'
    }
    print(f"Sending prompt....")
    
    # For Flan-T5, it's better to structure the prompt as an instruction
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 512,  # Adjust based on your needs
            "min_length": 100,
            "temperature": 0.5,  # Lower for more deterministic responses
            "num_return_sequences": 1
    }
               }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        try:
            response_json = response.json()
            # print("Response JSON:", response_json)
            
            return response_json[0]['generated_text']
                
        except Exception as e:
            print(f"Error processing response: {e}")
            print(f"Raw response: {response.text}")
            return f"Error: {str(e)}"
    else:
        error_msg = f"Error ({response.status_code}): {response.text}"
        print(error_msg)
        return error_msg

    
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
        context = retrieve_context(user_query, vector_store=vector_store, chunks=chunks)
        # print(f"Context: {context}\n\n")
        
        prompt = f"""Based on the following information, please answer the question thoroughly.
        INFORMATION:
        {context}

        QUESTION:
        {user_query}

        """
        
        print("Generating Response...\n\n")
        response = query_hf_llm(prompt=prompt)
        
        print(f"\n\nAnswer:{response}\n\n")
        
# RUN APP
if __name__ == "__main__":
    main()