import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from google.generativeai import GenerativeModel, configure
from neo4j import GraphDatabase

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

# Connect to the existing Pinecone index
index = pinecone_client.Index(PINECONE_INDEX)

# Constants
VECTOR_DIMENSION = 384
MAX_DECIMALS = 6  # Reduced decimal precision

# Configure Google Generative AI for natural language processing
API_KEY = os.getenv("GENAI_API_KEY")
configure(api_key=API_KEY)
model = GenerativeModel("gemini-1.5-flash")

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Schema information to be used by the AI model for Cypher queries
SCHEMA_INFO = """
The database schema contains:
- PLAYER nodes with properties: name, age, height, weight, number.
- TEAM nodes with properties: name.
- COACH nodes with properties: name, experience.
- Relationships: PLAYS_FOR between PLAYER and TEAM with property salary.
"""

# Define functions for Pinecone
def sanitize_vector(vector, max_decimals=MAX_DECIMALS):
    return [float(round(float(value), max_decimals)) for value in vector]

def is_valid_vector(vector):
    return np.all(np.isfinite(vector)) and not np.any(np.isnan(vector))

def normalize_vector(vector):
    vector = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.tolist()
    return (vector / norm).tolist()

def get_query_vector(query):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.cpu()
    model.eval()

    tokens = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
    
    query_embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy().astype(np.float32)
    query_embedding = normalize_vector(query_embedding)
    
    if not is_valid_vector(query_embedding):
        raise ValueError("The query vector contains invalid values.")
    
    return query_embedding

def nl_to_pinecone_query(nl_query):
    try:
        query_vector = get_query_vector(nl_query)
        if len(query_vector) != VECTOR_DIMENSION:
            raise ValueError(f"Query vector must have {VECTOR_DIMENSION} dimensions.")
        
        sanitized_query_vector = sanitize_vector(query_vector)
        return sanitized_query_vector
    except Exception as e:
        print(f"Error generating query vector: {e}")
        return None

def query_pinecone(query, top_k=5):
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

# Define functions for Neo4j
def query_neo4j(cypher_query):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run(cypher_query)
        return [record.data() for record in result]

def nl_to_neoquery(nl_query):
    prompt = f"""
    You are a Cypher query expert for Neo4j databases.
    {SCHEMA_INFO}
    Generate a valid Cypher query based on the following natural language query:
    "{nl_query}"
    Ensure the query matches the schema and uses exact labels and relationship names.
    Only return the Cypher query without explanation or code blocks.
    """
    response = model.generate_content(prompt)
    query = response.text.strip()
    query = query.replace("```", "").strip()
    if query.lower().startswith("cypher"):
        query = query[6:].strip()
    return query

def validate_query(cypher_query):
    corrections = {
        "Player": "PLAYER",
        "Team": "TEAM",
        "Coach": "COACH",
    }

    for wrong_label, correct_label in corrections.items():
        cypher_query = cypher_query.replace(wrong_label, correct_label)
    
    return cypher_query

def query_to_natural_language(results):
    if not results:
        return "No data found for your query."
    
    response = "Here are the results:\n"
    for idx, record in enumerate(results, start=1):
        response += f"{idx}. {record['p.name']} plays for {record['t.name']} with a salary of {record['r.salary']}\n"
    return response.strip()

# Main function to handle querying both databases
def answer():
    nl_query = input("Enter your query in natural language: ")

    try:
        # Query Pinecone
        pinecone_results = query_pinecone(nl_query, top_k=5)
        if pinecone_results:
            print("\n======== Query from Pinecone ===============================================================")
            print(pinecone_results)
        else:
            print("\nNo results found in Pinecone.")

        # Convert NL query to Cypher for Neo4j
        cypher_query = nl_to_neoquery(nl_query)
        print(f"\nGenerated Cypher Query for Neo4j: {cypher_query}")

        # Validate the query
        cypher_query = validate_query(cypher_query)
        print(f"Validated Cypher Query for Neo4j: {cypher_query}")

        # Query Neo4j
        neo4j_results = query_neo4j(cypher_query)
        if neo4j_results:
            print("\n============== Query from Neo4j ============================================================")
            print(query_to_natural_language(neo4j_results))
        else:
            print("\nNo results found in Neo4j.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    answer()
