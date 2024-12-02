import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np

# Load environment variables
load_dotenv()

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Initialize the Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def query_neo4j(query):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run(query)
        return [record.data() for record in result]

def natural_language_to_cypher(nl_query):
    # Convert the natural language query into Cypher format using sentence embeddings.
    # simulating the conversion based on a keyword match.
    embedding = model.encode(nl_query)
    # Example of converting NL to Cypher: we can define simple rules or use a more complex model.
    if "team" in nl_query:
        return "MATCH (n) RETURN n"
    elif "player" in nl_query:
        return "MATCH (n) RETURN n"
    else:
        return "MATCH (n) RETURN n"

if __name__ == "__main__":
    # Example natural language query
    nl_query = "Find all players and their teams"
    cypher_query = natural_language_to_cypher(nl_query)
    
    print(f"Cypher Query: {cypher_query}")
    data = query_neo4j(cypher_query)
    
    print("Query Results:", data)
