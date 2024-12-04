Workflow
Task 1: Data Preparation and Querying
1. Populate Neo4j
    Import the data into Neo4j using the players_and_teams.cypher file. This will populate your Neo4j database with player, team, coach, and game performance data.

2. Fetch Data
   Use the fetch_data.py script (under the src folder) to fetch data from Neo4j and process it:
   Transform and group the data.
   Save the grouped data to output/grouped_data.pkl.
3. Generate Sentences
   Use the create_sentences.py script (under the src folder) to generate sentences from the fetched data. These sentences are used to provide context for embedding.
   Save the chunked sentences to output/sentences_chunks.pkl.
4. Create Embeddings
   Use the generate_embedding.py script (under the src folder) to create embeddings for the generated sentences.
   Save the embeddings to output/embeddings.pkl.
5. Convert Embeddings to Vectors
   Use the convert_embeddings_to_vectors.py script (under the src folder) to convert the embeddings into vector format.
   Save the vectors to output/vectors.pkl.
6. Create and Upload Vectors to Pinecone
   Create an index and upload vectors to Pinecone using the create_index.py and upload_to_pinecone.py scripts.
7. Natural Language Queries
   Use NL-neo4j.py to query Neo4j based on natural language.
   Use NL-pinecone.py to query Pinecone with the same natural language prompt.
   Task 2: Implementing nl_to_neoquery and Displaying Results
1. Function to Convert NL to Cypher Query
   Implement the nl_to_neoquery() function to convert natural language input into a valid Cypher query for Neo4j.
   The function is located in NL-neo4j.py or query_integration.py.
2. Display Nodes and Edges
   Fetch the nodes and edges from Neo4j using the converted Cypher query.
   Use visualization-Neo4j-pinecone.py to visualize the graph data from Neo4j and Pinecone in a side-by-side comparison.
   The visualization includes the graph data and results from both Neo4j and Pinecone, displayed using libraries like NetworkX and Matplotlib.
3. Show Data alongside Results from Pinecone
   Use query_integration.py to query both databases (Neo4j and Pinecone) with the same natural language input.
   Display the results from both databases side by side (first Pinecone, then Neo4j).
4. Graph Visualization
   In visualization-Neo4j-pinecone.py, visualize the graph data from Neo4j and Pinecone together.
   Show nodes and edges in a visual form using a graph visualization library such as NetworkX.