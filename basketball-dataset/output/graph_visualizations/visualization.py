import networkx as nx
import matplotlib.pyplot as plt
import os

def visualize_graph(neo4j_results, filename="graph.png"):
    """
    Visualize the Neo4j results and save the graph as an image.
    
    Args:
        neo4j_results (list): List of nodes and relationships from Neo4j query.
        filename (str): Name of the output image file.
    """
    # Create a graph
    G = nx.Graph()
    
    # Add nodes and edges to the graph
    for result in neo4j_results:
        G.add_node(result['name'], team=result.get('team', 'Unknown'))
        if 'teammate' in result:
            G.add_edge(result['name'], result['teammate'])
    
    # Define output directory
    output_dir = os.path.join("output", "graph_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the graph as an image
    filepath = os.path.join(output_dir, filename)
    plt.figure(figsize=(12, 8))
    nx.draw(G, with_labels=True, node_size=1000, node_color='skyblue', font_size=10)
    plt.savefig(filepath)
    plt.close()
    print(f"Graph saved at {filepath}")
