import os
from fetch_data import fetch_data

def insert_data():
    """Insert sample data into Neo4j."""
    create_query = '''
        CREATE (russell:PLAYER {name:"Russell Westbrook", age:33, number:0, height:1.91, weight:91}),
               (lebron:PLAYER {name:"LeBron James", age:36, number:6, height:2.06, weight:113}),
               (curry:PLAYER {name:"Stephen Curry", age:35, number:30, height:1.91, weight:86}),
               (giannis:PLAYER {name:"Giannis Antetokounmpo", age:29, number:34, height:2.11, weight:110}),
               (kawhi:PLAYER {name:"Kawhi Leonard", age:32, number:2, height:2.01, weight:102}),
               (james:PLAYER {name:"James Harden", age:34, number:13, height:1.96, weight:100}),
               (kevin:PLAYER {name:"Kevin Durant", age:35, number:7, height:2.08, weight:109}),
               (luka:PLAYER {name:"Luka Dončić", age:25, number:77, height:2.01, weight:104}),
               (tatum:PLAYER {name:"Jayson Tatum", age:26, number:0, height:2.06, weight:95}),
               (butler:PLAYER {name:"Jimmy Butler", age:34, number:22, height:2.01, weight:104}),
               (bamba:PLAYER {name:"Mo Bamba", age:26, number:5, height:2.18, weight:104}),
               (bamba)-[:IS_TEAMMATE]->(butler),
               (james)-[:IS_TEAMMATE]->(kevin),
               (giannis)-[:IS_TEAMMATE]->(kawhi),
               (russell)-[:IS_TEAMMATE]->(lebron),
               (russell)-[:IS_TEAMMATE]->(curry),
               (lebron)-[:IS_TEAMMATE]->(kevin)
    '''
    return create_query

def read_cypher_file(file_path):
    """Reads the Cypher file and returns its contents."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return file.read()
    else:
        print(f"File not found: {file_path}")
        return ""

if __name__ == "__main__":
    # Test Neo4j connection with a simple query
    test_query = "MATCH (p:PLAYER) RETURN p LIMIT 10"
    test_data = fetch_data(test_query)
    
    if test_data:
        print("Neo4j connection is working.")
    else:
        print("Failed to fetch data from Neo4j. Check your connection or query.")
    
    # Insert data into Neo4j
    insert_query = insert_data()
    insert_data_result = fetch_data(insert_query)
    print("Data inserted successfully:", insert_data_result)

    # Example: Fetching data from a Cypher file
    cypher_file_path = "../data/players_and_teams.cypher"  # Adjust the path as needed
    cypher_query = read_cypher_file(cypher_file_path)
    print("Cypher query read from file:", cypher_query)
    
    if cypher_query:
        # Fetch data from Neo4j using the Cypher query from the file
        data = fetch_data(cypher_query)
        
        if data:
            print(f"Fetched {len(data)} records.")
            # Create a DataFrame from the result
            df = pd.DataFrame(data)
            print(f"DataFrame created with {len(df)} rows.")
            print(df.head())  # Print the first few rows of the DataFrame to check if it's correct
            
            # Output file path
            output_file = "../output/neo4j_data.csv"
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            print(f"Data saved to {output_file}.")
        else:
            print("No data returned from Neo4j. Please check your query.")
    else:
        print("No Cypher query found to execute.")
