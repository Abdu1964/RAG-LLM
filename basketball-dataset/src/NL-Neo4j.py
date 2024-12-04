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
- PLAYER nodes with properties: name, age, height, weight, number, salary.
- TEAM nodes with properties: name.
- COACH nodes with properties: name, experience.
- Relationships: PLAYS_FOR between PLAYER and TEAM with property salary.
"""

# Configure Google Generative AI for natural language processing
API_KEY = os.getenv("GENAI_API_KEY")
configure(api_key=API_KEY)
model = GenerativeModel("gemini-1.5-flash")

# Hardcoded questions and their corresponding Cypher queries
HARD_CODED_QUERIES = {
    "List all coaches who have coached multiple NBA teams": """
    MATCH (c:COACH)-[:COACHES]->(t:TEAM) 
    WITH c, count(t) AS teamCount 
    WHERE teamCount > 1 
    RETURN c.name
    """,
    "List all players who have played for more than one NBA team": """
    MATCH (p:PLAYER)-[:PLAYS_FOR]->(t:TEAM) 
    WITH p, count(t) AS teamCount 
    WHERE teamCount > 1 
    RETURN p.name
    """,
    "Who are the top 5 players in NBA based on salary": """
    MATCH (p:PLAYER)-[:PLAYS_FOR]->(t:TEAM)
    RETURN p.name, p.salary, t.name
    ORDER BY p.salary DESC
    LIMIT 5
    """,
    "Show the top 5 NBA players based on their salary and their associated teams": """
    MATCH (p:PLAYER)-[:PLAYS_FOR]->(t:TEAM)
    RETURN p.name, p.salary, t.name
    ORDER BY p.salary DESC
    LIMIT 5
    """,
    "Which NBA teams has LeBron James played for?": """
    MATCH (p:PLAYER {name: 'LeBron James'})-[:PLAYS_FOR]->(t:TEAM)
    RETURN t.name
    """,
    "What is the team composition of the Golden State Warriors in the 2019 season?": """
    MATCH (p:PLAYER)-[:PLAYS_FOR]->(t:TEAM {name: 'Golden State Warriors'})
    WHERE p.year = 2019
    RETURN p.name
    """,
    "Find the list of NBA players who have won MVP awards in the last 5 years": """
    MATCH (p:PLAYER)-[:HAS_AWARD]->(a:AWARD {name: 'MVP'})
    WHERE a.year >= (date().year - 5)
    RETURN p.name
    """,
    "Show me the game stats for the most recent NBA Finals": """
    MATCH (g:GAME {type: 'Finals'}) 
    WHERE g.date = (MAX(g.date))
    RETURN g.stats
    """,
    "Which NBA players have played for both the Boston Celtics and the Los Angeles Lakers": """
    MATCH (p:PLAYER)-[:PLAYS_FOR]->(t:TEAM)
    WHERE t.name IN ['Boston Celtics', 'Los Angeles Lakers']
    RETURN DISTINCT p.name
    """,
    "What are the career statistics for Kobe Bryant?": """
    MATCH (p:PLAYER {name: 'Kobe Bryant'})
    RETURN p.stats
    """
}

# Function to query Neo4j
def query_neo4j(cypher_query):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run(cypher_query)
        return [record.data() for record in result]

# Convert natural language to Cypher query
def nl_to_neoquery(nl_query):
    # Check if the NL query is hardcoded
    if nl_query in HARD_CODED_QUERIES:
        return HARD_CODED_QUERIES[nl_query]
    
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
