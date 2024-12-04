import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from google.generativeai import GenerativeModel, configure
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Configure Google Generative AI
API_KEY = os.getenv("GENAI_API_KEY")  # Make sure your API key is in the .env file
configure(api_key=API_KEY)
model = GenerativeModel("gemini-1.5-flash")

# Schema information to be used by the AI model
SCHEMA_INFO = """
The database schema contains:
- PLAYER nodes with properties: name, age, height, weight, number.
- TEAM nodes with properties: name.
- COACH nodes with properties: name, experience.
- Relationships: PLAYS_FOR between PLAYER and TEAM with property salary.
"""

def query_neo4j(cypher_query):
    """
    Executes a Cypher query on the Neo4j database and returns the results.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run(cypher_query)
        return [record.data() for record in result]

def nl_to_neoquery(nl_query):
    """
    Converts a natural language query to a Cypher query using Google Generative AI.
    """
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

    # Clean up any unnecessary text like markdown formatting or "cypher" tags.
    query = query.replace("```", "").strip()
    if query.lower().startswith("cypher"):
        query = query[6:].strip()

    return query

def validate_query(cypher_query):
    """
    Validates and fixes the generated Cypher query based on known schema.
    """
    corrections = {
        "Player": "PLAYER",
        "Team": "TEAM",
        "Coach": "COACH",
    }

    for wrong_label, correct_label in corrections.items():
        cypher_query = cypher_query.replace(wrong_label, correct_label)
    
    return cypher_query

def query_to_natural_language(results):
    """
    Converts the query results into a natural language response.
    """
    if not results:
        return "No data found for your query."
    
    response = "Here are the results:\n"
    for idx, record in enumerate(results, start=1):
        response += f"{idx}. {record}\n"
    return response.strip()

def answer():
    """
    Main function that processes the natural language query,
    converts it to Cypher, fetches the results, and returns the response.
    """
    nl_query = input("Enter your query in natural language: ")

    try:
        # Convert NL to Cypher query using the nl_to_neoquery function
        cypher_query = nl_to_neoquery(nl_query)
        print(f"Generated Cypher Query: {cypher_query}")

        # Validate the query
        cypher_query = validate_query(cypher_query)
        print(f"Validated Cypher Query: {cypher_query}")

        # Query Neo4j
        results = query_neo4j(cypher_query)

        # Convert results to natural language
        natural_language_response = query_to_natural_language(results)
        print(natural_language_response)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Call the answer function
    answer()
