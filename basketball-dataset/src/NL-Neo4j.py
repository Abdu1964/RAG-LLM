import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch
from neo4j import GraphDatabase
from google.generativeai import GenerativeModel, configure

# Load environment variables from .env file
load_dotenv()

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

# Configure Google Generative AI for natural language processing
API_KEY = os.getenv("GENAI_API_KEY")
configure(api_key=API_KEY)
model = GenerativeModel("gemini-1.5-flash")

# Function to query Neo4j
def query_neo4j(cypher_query):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run(cypher_query)
        return [record.data() for record in result]

# Convert natural language to Cypher query
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

# Validate Cypher query
def validate_query(cypher_query):
    corrections = {
        "Player": "PLAYER",
        "Team": "TEAM",
        "Coach": "COACH",
    }

    for wrong_label, correct_label in corrections.items():
        cypher_query = cypher_query.replace(wrong_label, correct_label)
    
    return cypher_query

# Main function to handle the query input and processing
def answer():
    nl_query = input("Enter your query in natural language: ")

    try:
        # Convert NL query to Cypher
        cypher_query = nl_to_neoquery(nl_query)
        print(f"\nGenerated Cypher Query for Neo4j: {cypher_query}")

        # Validate the query
        cypher_query = validate_query(cypher_query)
        print(f"Validated Cypher Query for Neo4j: {cypher_query}")

        # Query Neo4j
        neo4j_results = query_neo4j(cypher_query)
        if neo4j_results:
            print("\n============== Query from Neo4j ============================================================")
            for result in neo4j_results:
                print(result)
        else:
            print("\nNo results found in Neo4j.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    answer()
