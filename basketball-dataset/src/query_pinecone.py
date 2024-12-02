import pinecone
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone with API key and environment from .env
pinecone_client = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Initialize the Pinecone index
index_name = os.getenv("PINECONE_INDEX")
index = pinecone_client.Index(index_name)

# Set vector dimension and similarity metric (Cosine)
VECTOR_DIMENSION = 384
SIMILARITY_METRIC = "cosine"

def query_pinecone(query, top_k=5):
    # Ensure the query is converted to a vector (if it's not already)
    query_vector = get_query_vector(query)  # Function to get query vector

    # Debug: Print the query vector to verify its structure
    print("Query Vector:", query_vector)

    # Validate the query vector
    if not isinstance(query_vector, list):
        raise ValueError("The query vector must be a list or numpy array.")
    
    if len(query_vector) != VECTOR_DIMENSION:  # Check if the query vector has 384 dimensions
        raise ValueError(f"The query vector must have {VECTOR_DIMENSION} dimensions, but it has {len(query_vector)}.")

    # Perform the query to Pinecone
    try:
        results = index.query(queries=[query_vector], top_k=top_k, include_metadata=True)
        return results
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return None

def get_query_vector(query):
    # Convert the query into a vector (use my embedding model here)
    # Example: Generating a random vector
    query_vector = np.random.rand(VECTOR_DIMENSION).tolist()  # I can Replace with actual query vector logic
    return query_vector

if __name__ == "__main__":
    # Sample query for testing
    query = "Who are the top players in the NBA?"
    top_k = 5

    # Query Pinecone and print results
    results = query_pinecone(query, top_k)
    
    if results:
        print("Query Results:", results)
