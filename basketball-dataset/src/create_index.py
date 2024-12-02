from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import os

# Load environment variables from .env
load_dotenv()

# Neo4j credentials
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Pinecone credentials
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")



# Check if required variables are loaded
if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, PINECONE_API_KEY, PINECONE_ENV, PINECONE_CLOUD]):
    raise ValueError("One or more environment variables are missing. Please check the .env file.")

# Initialize Neo4j driver
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Initialize Pinecone client
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# Function to create an index in Pinecone
def create_index(index_name, dimension):
    existing_indexes = pinecone_client.list_indexes().names()
    if index_name not in existing_indexes:
        pinecone_client.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_ENV)
        )
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")

#  values
index_name = "pinecone-index-vector-storage"
dimension = 384

# Create the Pinecone index
create_index(index_name, dimension)

# Close the Neo4j driver when done
neo4j_driver.close()
