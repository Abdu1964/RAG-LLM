from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from collections import defaultdict
import pickle

# Load environment variables
load_dotenv()

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Function to read the Cypher query from a file
def read_cypher_file(file_path):
    with open(file_path, 'r') as file:
        cypher_query = file.read()
    return cypher_query

# Function to execute Cypher query
def execute_cypher_query(query, commit=False):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            print("Executing query:", query)  # Debug: print the query
            if commit:
                with session.begin_transaction() as tx:
                    tx.run(query)
                    print("Transaction committed!")  # Debug message
            else:
                result = session.run(query)
                data = [record.data() for record in result]
                print("================================ 1. FETCH DATA ====================================================")
                print("Fetched Data:", data)  # Debug: print fetched data
                print("====================================================================================================")
                if not data:
                    print("No data was returned from the query.")
                return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
    finally:
        driver.close()

# Path to the 'players_and_teams.cypher' file
cypher_file_path = os.path.join(os.path.dirname(__file__), '../data/players_and_teams.cypher')

# Read the Cypher query from the file
cypher_query = read_cypher_file(cypher_file_path)

# Call the function to execute the Cypher query (for populating the database)
execute_cypher_query(cypher_query, commit=True)

# Fetch all data from the database without any LIMIT
fetch_query = """
MATCH (n)-[r]->(m)
RETURN n AS start_node, type(r) AS relationship_type, m AS end_node
"""
fetched_data = execute_cypher_query(fetch_query)

# Check if the fetched data is None or empty
if fetched_data is None or len(fetched_data) == 0:
    print("No data fetched or there was an error.")
else:
    # Function to transform and group data by context (team, relationship type, etc.)
    def transform_and_group_data(fetched_data):
        grouped_data = {}
        for record in fetched_data:
            if 'start_node' in record and 'relationship_type' in record and 'end_node' in record:
                start_node = record['start_node']
                relationship_type = record['relationship_type']
                end_node = record['end_node']
                
                # Initialize a relationship type if it doesn't exist
                if relationship_type not in grouped_data:
                    grouped_data[relationship_type] = {}
                
                # Add the start_node and end_node relationship
                grouped_data[relationship_type][str(start_node)] = str(end_node)
            else:
                print(f"Missing data in record: {record}")
        return grouped_data

    # Call the function to group the fetched data
    grouped_data = transform_and_group_data(fetched_data)

    # Print the grouped data to check if the transformation is successful
    print("======================================================================================================")
    print("====================================== 2. GROUPED DATA ================================================")
    print("======================================================================================================")
    print("Grouped Data:", grouped_data)

    # Save the grouped data to a pickle file
    output_path = os.path.join(os.path.dirname(__file__), '../output/grouped_data.pkl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as file:
        pickle.dump(grouped_data, file)
    print(f"Grouped data saved to {output_path}")
