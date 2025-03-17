import os
import fitz  # PyMuPDF for PDF extraction
from dotenv import load_dotenv
import requests
import faiss # FAISS - Facebook AI Similarity Search
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore

load_dotenv() # load the env content from the .env file

api_key = os.getenv('HF_API_KEY')
os.environ['HF_API_KEY'] = api_key

# === CONFIGURATION ===
HF_API_KEY = api_key  # Set your Hugging Face API key here
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"  # Can change inference model
PDF_DIR = "data"  # Directory containing PDF files


# === LOADING THE PDF DATA FROM THE DIRECTORIES INTO A LIST OF DOCUMENT DATA ===

# Takes in the directory path that is in the working folder holding the pdf documents as "str"(PDF_DIR), as a parameter.
# Return a list of "Document" object instances containing text data from all the pdfs and their meta data.
def load_pdfs_from_directory(pdf_dir):
    
    # Initializes an array (document) to store information from each document.
    documents = []
    
    # The loop runs through each file in the directory and checks whether the file is a pdf or not.
    for file_name in os.listdir(pdf_dir): 
        
        # If the file is a pdf we establish the file path
        if file_name.endswith(".pdf"):
            
            # file_path established by joining the name of the file with the .pdf for opening the file with a pdf opener library called **fitz**(alias for PyMuPdf).
            file_path = os.path.join(pdf_dir, file_name)
            
            with fitz.open(file_path) as doc: # "as doc" returns a document object that allows you to interact with the pdf
                text = "\n".join([page.get_text() for page in doc]) # The doc object is iterable through page objects, each iterable is a page object.
                # "\n".join is used to join text from each page of the pdf with a newline in between.
                # "text" now contains all the text from a single pdf document.
                
                # Document class is used to store structured text data along with meta data(dict storing dditional info like filename)
                # page_content :  the extracted text from a PDF
                documents.append(Document(page_content=text, metadata={"source":file_name}))
                
    # documents array stores instances of the "Document" class containing pdf content as text and metadata as file name.
    return documents

# === GETTING THE EMBEDDINGS FROM THE HUGGING FACE API ===

# The function takes in "texts" as an array of strings and returns an array of embeddings for each text in "texts".
def get_hf_embeddings(texts):
    
    # url for specifying which embedding model to use (API Endpoint)
    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"
    
    # The "headers" dict includes an authorization token i.e., Hugging Face API key
    # which is required to authenticate the request with the Hugging Face API.
    headers = {
        'Authorization': f'Bearer {HF_API_KEY}'
        }
    
    # API request a POST request is sent to Hugging Face API
    response = requests.post(url, headers=headers, json={"inputs":texts})
    
    # Response Handling if API request is successful that means "response.status_code == 200"
    # the function "response.json()" returns a JSON response which contains the embeddings for text inputs. 
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching embeddings: {response.json()}")
    

# === CREATING A VECTOR DATABASE USING FAISS WITH THE DOCUMENT TEXT DATA ===

# The function takes in the documents list containing the "Document" instances as a parameter    
def create_faiss_vectorstore(documents):
    
    # Defining an object "text_splitter" for the class "RecursiveCharacterTextSplitter" (Uses a recursive approach to break text at the most natural boundaries (e.g., sentences → words → characters).)
    # which helps to create chunks of data for easy processing with a chunk overlap.
    # chunk_size: each chunk has 512 characters max;
    # chunk_overlap: Adjacent chunks will have max 50 overlapping character to ensure important context is not lost between chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    
    # "split_documents" takes the documents list and splits each "Document" in the list into smaller chunks based on the above function.
    # "chunks" now stores a list of Document objects(chunk) which now stores only text of 512 characters or less and its meta data.
    chunks = text_splitter.split_documents(documents)
    
    # "texts" array stores just the page content of the chunks as strings.
    texts = [chunk.page_content for chunk in chunks]
    
    # "metadatas" array stores document(chunk) id for each text in "texts" array
    metadatas = [{"source": f"doc_{i}"} for i in range(len(chunks))]

    # get_hf_embeddings functions takes in "texts" array to convert them into vectors and get the embeddings.
    embeddings = get_hf_embeddings(texts)
    # print(type(embeddings[0]))

    # "diimension" stores the size of an embedding for any text in the "texts" array
    dimension = len(embeddings[0]) # Get embeddings size, typically 384 for "Sentence Transformers Embeddings Model" 
    
    # "IndexFlatL2 creates a FAISS index for fast similarity search using the L2(Euclidean) Distance Metric"
    faiss_index = faiss.IndexFlatL2(dimension)
    
    # Add the embeddings (converted to float32 numpy array) to the FAISS index
    faiss_index.add(np.array(embeddings, dtype=np.float32))
    
    # Create docstore (mapping between document ID and actual content)
    
    # parsing the index "i" into string because "InMemoryDocstore" only expects inputs as strings and not integers
    # "documents_dict" stores a mapping of the index and documents objects w.r.t. "texts" array containing chunks and their metadata
    # Eg: { "0" : Document(page_content:"chunk1", metadata={"source": "doc_0"}),
    #       "1" : Document(page_content:"chunk2", metadata={"source": "doc_1"})
    # }
    documents_dict = {str(i): Document(page_content=texts[i], metadata=metadatas[i]) for i in range(len(texts))}
    
    # Creates an in-memory document store that
    # maps document IDs to their corresponding document objects.
    
    # A specialized abstraction designed to store and retrieve documents in the context of vector databases.
    # It enforces a specific structure(e.g., Document objects with page_content and metadata) and provides methods optimized for document retrieval.
    
    # InMemoryDocstore is tightly integrated with LangChain's ecosystem, 
    # making it compatible with other components like FAISS and vector stores.
    docstore = InMemoryDocstore(documents_dict)

    # Create index-to-document mapping
    # Mapping the FAISS index position to the corresponding document ID
    # This mapping is used to link the FAISS index to the document store.
    index_to_docstore_id = {i: str(i) for i in range(len(texts))}
    
    # Wrap FAISS index with LangChain's FAISS
    vector_store = FAISS(
        index=faiss_index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        embedding_function=lambda x: get_hf_embeddings(x)  # Query embedding function
    )
    
    # returning the FAISS-based vector database (vector_store) and the chunks array.
    return vector_store, chunks

# This funciton takes in the user-query, FAISS based vector database, and chunks as input
# and returns top 3 similar retrieved text joined together as output string.
def retrieve_context(query, vector_store, chunks):
    
    # Since we are sending the query in as a list object, we will get the embeddings in a list objects as well,
    # thats why we select the first index below to avoid sending a 2D array to vector_atore.index.search
    query_embedding = get_hf_embeddings([query])[0]
    
    # np.array([query_embedding], dtype=np.float32) converts query_embedding of shape (384,)
    # into a 2D array of shape (1, 384), which is what FAISS expects for a single query.
    
    # (n, 384) => n is number of queries of vector size 384.
    
    # k=3 specifies the top 3 most similar embeddings (based on L2 distance) should be retrieved.
    
    # the vectors search returns an array of distances between query embedding and retrieved embeddings
    # and an array of indices corresponding to the most similar embeddings in the FAISS index.
    
    distances, indices = vector_store.index.search(np.array([query_embedding], dtype=np.float32), k=3)
    
    # Finally the most similar chunks are retrieved using the indices array.
    retrieved_texts = [chunks[i].page_content for i in indices[0] if i < len(chunks)]
    
    # The retieved text chunks are joined together wil double newline(\n\n) to form a single string
    # The combined string is returned as the context for the query.
    return "\n\n".join(retrieved_texts)

# === GETTING THE ANSWER GIVEN A PROMPT FROM THE HUGGING FACE API ===
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
            
            # Check how the response format is first and then return accordingly
            # Different models have different return formats for json.
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
        
        # The prompt contains the context and the query and levergaes an LLM already pretrained to answer the given query.
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