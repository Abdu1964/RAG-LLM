import os
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from pinecone import Pinecone
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

# Pinecone and Neo4j configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Constants
VECTOR_DIMENSION = 384

# Pinecone Initialization
pinecone = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX not in pinecone.list_indexes().names():
    pinecone.create_index(
        name=PINECONE_INDEX,
        dimension=VECTOR_DIMENSION,
        metric="cosine"
    )
index = pinecone.Index(PINECONE_INDEX)

# Sentence Embedding Model Initialization
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name)

# Neo4j Driver
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Function: Normalize and Validate Vector
def normalize_vector(vector):
    vector = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    return (vector / norm).tolist() if norm > 0 else vector.tolist()

def is_valid_vector(vector):
    return np.all(np.isfinite(vector)) and not np.any(np.isnan(vector))

# Function: Generate Query Embedding
def generate_embedding(query):
    tokens = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = embedding_model(**tokens)
    embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy()
    embedding = normalize_vector(embedding)
    if not is_valid_vector(embedding):
        raise ValueError("Invalid vector generated.")
    return embedding

# Function: Query Pinecone
def query_pinecone(nl_query, top_k=5):
    query_vector = generate_embedding(nl_query)
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return results

# Function: Query Neo4j
def query_neo4j(cypher_query):
    with neo4j_driver.session() as session:
        return [record.data() for record in session.run(cypher_query)]

# Function: Visualize Neo4j Graph
def visualize_neo4j_graph(nodes, relationships):
    graph = nx.DiGraph()
    
    # Add nodes
    for node in nodes:
        graph.add_node(node["id"], label=node["label"])
    
    # Add edges
    for rel in relationships:
        graph.add_edge(rel["start"], rel["end"], label=rel["type"])
    
    # Draw graph
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=5000, node_color="skyblue", font_size=10, font_weight="bold")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, "label"))
    plt.show()

# Function: Visualize Vector Relationships
def visualize_vector_graph(vectors, labels):
    graph = nx.DiGraph()

    # Add nodes and edges
    for i, vector in enumerate(vectors):
        graph.add_node(i, label=labels[i])
        for j, related_vector in enumerate(vectors):
            similarity = np.dot(vector, related_vector)
            if similarity > 0.8:  # Example threshold for strong relationships
                graph.add_edge(i, j, weight=similarity)

    # Draw graph
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=5000, node_color="lightgreen", font_size=10, font_weight="bold")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, "weight"))
    plt.show()

# Function: Process Query
def process_query(nl_query):
    print(f"Processing query: {nl_query}\n")

    # Pinecone Query
    print("Querying Pinecone...")
    pinecone_results = query_pinecone(nl_query, top_k=5)
    print("Pinecone Results:")
    for result in pinecone_results.matches:
        print(result)

    # Neo4j Query
    print("\nGenerating Cypher query for Neo4j...")
    cypher_query = f"MATCH (n) WHERE n.name CONTAINS '{nl_query}' RETURN n"
    print(f"Generated Cypher Query: {cypher_query}")

    print("Querying Neo4j...")
    neo4j_results = query_neo4j(cypher_query)
    print("Neo4j Results:")
    for result in neo4j_results:
        print(result)

    # Visualizations
    print("\nVisualizing Neo4j Graph...")
    nodes = [{"id": "1", "label": "Node1"}, {"id": "2", "label": "Node2"}]
    relationships = [{"start": "1", "end": "2", "type": "RELATES_TO"}]
    visualize_neo4j_graph(nodes, relationships)

    print("\nVisualizing Vector Relationships...")
    sample_vectors = [np.random.rand(384) for _ in range(5)]
    sample_labels = [f"Node{i}" for i in range(1, 6)]
    visualize_vector_graph(sample_vectors, sample_labels)

# Main Execution
if __name__ == "__main__":
    query = input("Enter your natural language query: ")
    process_query(query)

