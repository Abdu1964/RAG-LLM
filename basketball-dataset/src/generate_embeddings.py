import os
import pickle
from transformers import AutoTokenizer, AutoModel
import torch

# Load the embedding model
def load_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

# Generate embeddings for chunks
def generate_embeddings(chunks, tokenizer, model):
    embeddings = []
    for chunk in chunks:
        chunk_text = " ".join(chunk)
        tokens = tokenizer(chunk_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            output = model(**tokens)
        chunk_embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(chunk_embedding)
    return embeddings

if __name__ == "__main__":
    # Load the chunks
    chunks_file_path = "../output/sentences_chunks.pkl"
    with open(chunks_file_path, 'rb') as file:
        chunks = pickle.load(file)

    # Load model
    tokenizer, model = load_model()

    # Generate embeddings
    embeddings = generate_embeddings(chunks, tokenizer, model)

    # Print embeddings before saving
    print("================================== Embeddings Before Saving ==================================")
    for i, embedding in enumerate(embeddings):
        print(f"Embedding {i + 1}: {embedding}")
    print("=============================================================================================")

    # Save embeddings to a file
    embeddings_file_path = "../output/embeddings.pkl"
    with open(embeddings_file_path, 'wb') as file:
        pickle.dump(embeddings, file)

    print(f"Embeddings saved to {embeddings_file_path}")
