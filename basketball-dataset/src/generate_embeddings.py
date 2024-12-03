import os
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Normalize the embedding vector
def normalize_vector(vector):
    """
    Normalizes a vector to have values between -1 and 1.
    Args:
        vector (list): Input vector.
    Returns:
        list: Normalized vector.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:  # Avoid division by zero
        return vector
    return (np.array(vector) / norm).tolist()

# Load the embedding model
def load_model():
    """
    Load the sentence transformer model and tokenizer.
    Returns:
        tokenizer: Tokenizer for the model.
        model: Pre-trained sentence transformer model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

# Generate embeddings for chunks
def generate_embeddings(chunks, tokenizer, model):
    """
    Generate embeddings for text chunks.
    Args:
        chunks (list of list of str): List of text chunks to embed.
        tokenizer: Pre-trained tokenizer.
        model: Pre-trained model.
    Returns:
        embeddings (list of list of float): List of embedding vectors as lists of floats.
    """
    embeddings = []
    for chunk in chunks:
        chunk_text = " ".join(chunk)  # Combine sentences in the chunk
        tokens = tokenizer(chunk_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            output = model(**tokens)
        # Convert the tensor to a list of floats
        chunk_embedding = output.last_hidden_state.mean(dim=1).squeeze().tolist()
        # Normalize the embedding
        chunk_embedding = normalize_vector(chunk_embedding)
        embeddings.append(chunk_embedding)
    return embeddings

if __name__ == "__main__":
    # Path to the input file containing text chunks
    chunks_file_path = "../output/sentences_chunks.pkl"
    
    # Path to save the embeddings
    embeddings_file_path = "../output/embeddings.pkl"
    
    # Load the text chunks
    print("Loading text chunks...")
    with open(chunks_file_path, 'rb') as file:
        chunks = pickle.load(file)
    print(f"Loaded {len(chunks)} chunks.")

    # Load the model and tokenizer
    print("Loading the model and tokenizer...")
    tokenizer, model = load_model()

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(chunks, tokenizer, model)
    print("Embeddings generated successfully.")

    # Print embeddings for debugging (optional)
    print("================================== Embeddings Before Saving ==================================")
    for i, embedding in enumerate(embeddings[:5]):  # Show only the first 5 embeddings for brevity
        print(f"Embedding {i + 1}: {embedding[:10]}... (truncated)")  # Show only the first 10 dimensions
    print("=============================================================================================")

    # Save embeddings to a file
    print(f"Saving embeddings to {embeddings_file_path}...")
    with open(embeddings_file_path, 'wb') as file:
        pickle.dump(embeddings, file)
    print(f"Embeddings saved successfully to {embeddings_file_path}.")
