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
with open("../output/vectors.pkl", "rb") as file:
    embeddings = pickle.load(file)

with open("../output/sentences_chunks.pkl", "rb") as file:
    sentences = pickle.load(file)

print(len(sentences), "this is sentences", len(embeddings))

# Initialize Pinecone client
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# Connect to the Pinecone index
index = pinecone_client.Index("pinecone-index-vector-storage-2")

# Prepare the data for upload (embedding vectors + metadata)
# Assuming embeddings are a list of vectors and sentences are a list of corresponding metadata
upsert_data = []
for i, (embedding, sentence) in enumerate(zip(embeddings, sentences)):
    # Creating a unique ID for each embedding
    player_id = f"player_{i}"
    # Adding metadata (e.g., sentence text)
    metadata = {"sentence": sentence}
    upsert_data.append((player_id, embedding, metadata))

# Upload the data to Pinecone
index.upsert(vectors=upsert_data)

print("Data with sentences uploaded to Pinecone successfully.")
