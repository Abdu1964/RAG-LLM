from neo4j import GraphDatabase
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def fetch_data(query):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run(query)
        data = [record.data() for record in result]
    driver.close()
    return data

if __name__ == "__main__":
    cypher_query = """
    MATCH (n:Person)-[:KNOWS]->(m:Person)
    RETURN n.name AS source, m.name AS target
    """
    data = fetch_data(cypher_query)
    df = pd.DataFrame(data)
    output_file = "output/neo4j_data.csv"
    df.to_csv(output_file, index=False)
    print(f"Data fetched and saved to {output_file}.")
