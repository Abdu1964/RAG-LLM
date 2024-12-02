from dotenv import load_dotenv
from pinecone import Pinecone
import os
import pickle

# Load environment variables from .env
load_dotenv()

# Pinecone credentials
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")

# Load the embeddings from the pickle file
with open("../output/embeddings.pkl", "rb") as file:
    embeddings = pickle.load(file)

# Initialize Pinecone client
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# Connect to the Pinecone index
index = pinecone_client.Index("pinecone-index-vector-storage")

# Prepare the data for upload (embedding vectors + metadata)
# Assuming my embeddings are in a list of tuples [(id, vector), ...]
upsert_data = []
for i, embedding in enumerate(embeddings):
    # Creating a unique ID for each embedding (e.g., based on player name)
    player_id = f"player_{i}"  # Customize this as needed
    upsert_data.append((player_id, embedding))

# Upload the data to Pinecone
index.upsert(vectors=upsert_data)

print("Data uploaded to Pinecone successfully.")

