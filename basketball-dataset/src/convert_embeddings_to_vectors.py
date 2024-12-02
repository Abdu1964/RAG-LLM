import pickle
import numpy as np

# Load the embeddings
def load_embeddings(embeddings_file_path):
    with open(embeddings_file_path, 'rb') as file:
        embeddings = pickle.load(file)
    return embeddings

# Convert embeddings to vectors 
def convert_to_vectors(embeddings):
    vectors = [np.array(embedding) for embedding in embeddings]
    return vectors

if __name__ == "__main__":
    # Load embeddings
    embeddings_file_path = "../output/embeddings.pkl"
    embeddings = load_embeddings(embeddings_file_path)

    # Convert embeddings to vectors
    vectors = convert_to_vectors(embeddings)

    # Print vectors before saving
    print("================================== Vectors Before Saving ==================================")
    for i, vector in enumerate(vectors):
        print(f"Vector {i + 1}: {vector}")
    print("=========================================================================================")

    # Save vectors to a file for further use
    vectors_file_path = "../output/vectors.pkl"
    with open(vectors_file_path, 'wb') as file:
        pickle.dump(vectors, file)

    print(f"Vectors saved to {vectors_file_path}")
