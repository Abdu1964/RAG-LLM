from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from collections import defaultdict

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
    # Create a driver inside the function
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            print("Executing query:", query)  # Debugging: print the query
            
            # Use a write transaction if commit=True
            if commit:
                # Start a transaction for writing data
                with session.begin_transaction() as tx:
                    tx.run(query)  # Run the write query
                    print("Transaction committed!")  # Debug message
            else:
                # For read-only queries, just run and fetch results
                result = session.run(query)
                data = [record.data() for record in result]
                
                # Print the fetched data to check if it is empty or not
                print("Fetched Data:", data)  # Debug: print the data fetched
                if not data:
                    print("No data was returned from the query.")
                return data
            
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
    finally:
        driver.close()

# Set the correct path to the 'players_and_teams.cypher' file
cypher_file_path = os.path.join(os.path.dirname(__file__), '../data/players_and_teams.cypher')

# Read the Cypher query from the file
cypher_query = read_cypher_file(cypher_file_path)

# Call the function to execute the Cypher query (for populating the database)
execute_cypher_query(cypher_query, commit=True)  # Change to commit=True to save changes

# Fetch all data from the database without any LIMIT
fetch_query = """
MATCH (n)-[r]->(m)
RETURN n AS start_node, type(r) AS relationship_type, m AS end_node
"""

# Fetch the data (no LIMIT, fetch all data)
fetched_data = execute_cypher_query(fetch_query)

# Check if the fetched data is None or empty
if fetched_data is None or len(fetched_data) == 0:
    print("No data fetched or there was an error.")
else:
    # Function to transform and group data by context (team, relationship type, etc.)
    def transform_and_group_data(fetched_data):
        # Initialize defaultdict to group data by relationship type, and nodes
        grouped_data = defaultdict(lambda: defaultdict(list))
        
        for record in fetched_data:
            # Ensure that relevant fields exist in the record
            if 'start_node' in record and 'relationship_type' in record and 'end_node' in record:
                start_node = record['start_node']
                relationship_type = record['relationship_type']
                end_node = record['end_node']
                
                # Grouping data based on relationship type
                grouped_data[relationship_type][str(start_node)] = str(end_node)
            else:
                print(f"Missing data in record: {record}")
        
        return grouped_data

    # Call the function to group the fetched data
    grouped_data = transform_and_group_data(fetched_data)

    # Print the grouped data to check if the transformation is successful
    print("Grouped Data:", grouped_data)
