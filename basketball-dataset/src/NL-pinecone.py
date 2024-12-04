import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from google.generativeai import GenerativeModel, configure

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone instance
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Get Pinecone environment and index from environment variables
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Ensure the environment and index are set
if not PINECONE_ENVIRONMENT or not PINECONE_INDEX:
    raise ValueError("PINECONE_ENVIRONMENT and PINECONE_INDEX must be set in the .env file.")

# Check if the index exists; if not, create it
if PINECONE_INDEX not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=PINECONE_INDEX,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=PINECONE_ENVIRONMENT
        )
    )

# Connect to the existing index
index = pinecone_client.Index(PINECONE_INDEX)

# Constants
VECTOR_DIMENSION = 384
MAX_DECIMALS = 6  # Reduced decimal precision

# Configure Google Generative AI for natural language processing
API_KEY = os.getenv("GENAI_API_KEY")
configure(api_key=API_KEY)
model = GenerativeModel("gemini-1.5-flash")

def sanitize_vector(vector, max_decimals=MAX_DECIMALS):
    """Ensures vector values are rounded to the specified decimal precision and converted to float32."""
    return [float(round(float(value), max_decimals)) for value in vector]

def is_valid_vector(vector):
    """Checks if a vector contains valid values (no NaN, Inf, or unexpected values)."""
    return np.all(np.isfinite(vector)) and not np.any(np.isnan(vector))

def normalize_vector(vector):
    """Normalizes a vector so that its magnitude is 1."""
    vector = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.tolist()
    return (vector / norm).tolist()

def get_query_vector(query):
    """Converts the query into a vector using an embedding model."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Move model to CPU and set to eval mode
    model.cpu()
    model.eval()

    tokens = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        output = model(**tokens)
    
    # Convert to numpy, then to list
    query_embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy().astype(np.float32)
    
    # Normalize and convert to list
    query_embedding = normalize_vector(query_embedding)
    
    # Validate the vector
    if not is_valid_vector(query_embedding):
        raise ValueError("The query vector contains invalid values.")
    
    return query_embedding

def nl_to_pinecone_query(nl_query):
    """Generates a Pinecone-compatible query vector from a natural language query."""
    try:
        query_vector = get_query_vector(nl_query)
        # Validate vector dimension
        if len(query_vector) != VECTOR_DIMENSION:
            raise ValueError(f"Query vector must have {VECTOR_DIMENSION} dimensions, but has {len(query_vector)}.")
        
        sanitized_query_vector = sanitize_vector(query_vector)
        return sanitized_query_vector
    except Exception as e:
        print(f"Error generating query vector: {e}")
        return None

def query_pinecone(query, top_k=5):
    """Queries Pinecone with a natural language query after converting it to a vector."""
    try:
        sanitized_query_vector = nl_to_pinecone_query(query)
        if not sanitized_query_vector:
            raise ValueError("Failed to generate a valid query vector.")

        results = index.query(
            vector=sanitized_query_vector,
            top_k=top_k,
            include_metadata=True
        )
        return results
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return None

def answer():
    """Main function to process the natural language query and fetch Pinecone results."""
    nl_query = input("Enter your query in natural language: ")

    try:
        # Query Pinecone
        results = query_pinecone(nl_query, top_k=5)

        if results:
            print("Query Results:", results)
        else:
            print("No results found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    answer()
