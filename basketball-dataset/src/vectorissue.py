import torch
import numpy as np

# Normalize the query embedding vector
def normalize_vector(vec):
    """
    Normalize the vector to unit length.
    Args:
        vec (list or np.array): The vector to normalize.
    Returns:
        np.array: The normalized vector.
    """
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

# Function to remove or replace problematic values in the vector
def clean_vector(vector, problematic_index=None, replace_value=0.0):
    """
    Clean the vector by removing or replacing problematic values.
    
    Args:
        vector (list or np.array): The vector to clean.
        problematic_index (int): The index of the problematic value to remove/replace.
        replace_value (float): The value to replace the problematic one with. Default is 0.
        
    Returns:
        list: The cleaned vector.
    """
    if problematic_index is not None:
        # Replace the problematic index with a default value (0 or another chosen value)
        vector[problematic_index] = replace_value
    return vector

def get_query_vector(query, tokenizer, model, problematic_index=None):
    """
    Converts a natural language query into an embedding vector, while handling problematic values.
    
    Args:
        query (str): The natural language query.
        tokenizer: Pre-trained tokenizer.
        model: Pre-trained model.
        problematic_index (int): The index to remove/replace if an invalid value is found.
        
    Returns:
        list: Cleaned, normalized query embedding vector.
    """
    tokens = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
    query_embedding = output.last_hidden_state.mean(dim=1).squeeze().tolist()

    # Clean the vector if a problematic index is specified
    if problematic_index is not None:
        query_embedding = clean_vector(query_embedding, problematic_index)

    # Normalize the query embedding vector
    query_embedding = normalize_vector(query_embedding)

    return query_embedding

def query_pinecone(query, tokenizer, model, top_k=5, problematic_index=None):
    """
    Queries Pinecone using a normalized query vector.
    
    Args:
        query (str): The natural language query.
        tokenizer: Pre-trained tokenizer.
        model: Pre-trained model.
        top_k (int): The number of results to return.
        problematic_index (int): Index of the problematic value to clean/replace.
        
    Returns:
        dict: The results from Pinecone.
    """
    # Get the query vector
    query_vector = get_query_vector(query, tokenizer, model, problematic_index)
    print(f"Query Vector: {query_vector} ...")
    
    # Insert your Pinecone query logic here (example shown for simplicity)
    # For example, if using Pinecone's Python SDK:
    # response = pinecone.query(
    #     vector=query_vector,
    #     top_k=top_k,
    #     include_metadata=True
    # )
    
    # Placeholder response
    response = {"matches": [{"id": "player1", "score": 0.95, "metadata": {"name": "LeBron James"}},
                            {"id": "player2", "score": 0.92, "metadata": {"name": "Stephen Curry"}}]}
    
    return response

if __name__ == "__main__":
    # Load the model and tokenizer
    print("Loading the embedding model...")
    tokenizer, model = load_model()  # Implement load_model based on your specific model
    
    # Define a natural language query
    query = "Who are the top players in the NBA?"
    
    # Specify the problematic index to clean (e.g., 374 or None if not needed)
    problematic_index = 374

    # Query Pinecone
    print(f"Querying Pinecone for: {query}")
    results = query_pinecone(query, tokenizer, model, top_k=5, problematic_index=problematic_index)

    # Display results
    if results:
        print("\nQuery Results:")
        for match in results['matches']:
            print(f"ID: {match['id']}, Score: {match['score']}, Metadata: {match.get('metadata', {})}")
    else:
        print("No results found.")
