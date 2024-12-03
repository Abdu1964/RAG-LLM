import os
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModel
import torch

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone instance
pinecone_client = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

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

def sanitize_vector(vector, max_decimals=MAX_DECIMALS):
    """
    Ensures vector values are rounded to the specified decimal precision and converted to float32.
    """
    return [float(round(float(value), max_decimals)) for value in vector]

def is_valid_vector(vector):
    """
    Checks if a vector contains valid values (no NaN, Inf, or unexpected values).
    """
    return np.all(np.isfinite(vector)) and not np.any(np.isnan(vector))

def normalize_vector(vector):
    """
    Normalizes a vector so that its magnitude is 1.
    """
    vector = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.tolist()
    return (vector / norm).tolist()

def get_query_vector(query):
    """
    Converts the query into a vector using an embedding model.
    """
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

def query_pinecone(query, top_k=5):
    """
    Queries Pinecone with a sanitized query vector.
    """
    try:
        query_vector = get_query_vector(query)

        # Validate vector dimension
        if len(query_vector) != VECTOR_DIMENSION:
            raise ValueError(f"Query vector must have {VECTOR_DIMENSION} dimensions, but has {len(query_vector)}.")

        # Sanitize the query vector
        sanitized_query_vector = sanitize_vector(query_vector)

        # Perform the query with vector as a list
        results = index.query(
            vector=sanitized_query_vector,  # Changed from queries to vector
            top_k=top_k,
            include_metadata=True
        )
        return results
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return None

if __name__ == "__main__":
    query = "Who are the top players in the NBA?"
    top_k = 5

    results = query_pinecone(query, top_k)

    if results:
        print("Query Results:", results)
    else:
        print("No results found.")