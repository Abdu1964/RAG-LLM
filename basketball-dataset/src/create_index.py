from dotenv import load_dotenv
from pinecone import Pinecone
import os
import pickle
from sentence_transformers import SentenceTransformer  # A popular model for embedding text

# Load environment variables
load_dotenv()

# Pinecone credentials
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")

# Initialize Pinecone client
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone_client.Index("pinecone-index-vector-storage")

# Load the Sentence Transformer model (or use any other model to embed your query)
model = SentenceTransformer('all-MiniLM-L6-v2')  # This is just an example model

# Function to query Pinecone
def query_pinecone(query, top_k=5):
    # Transform the query into a vector
    query_vector = model.encode([query])[0]  # Get the vector for the query

    # Query Pinecone for the top_k most similar vectors
    results = index.query(queries=[query_vector], top_k=top_k)

    return results

# Example: Natural language query
query = "Who is the highest paid player in the team?"

# Query Pinecone with the NL query
results = query_pinecone(query)

# Display the results
print("Query Results:")
for match in results['matches']:
    print(f"ID: {match['id']}, Score: {match['score']}")
