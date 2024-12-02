import os
import pickle

# Load the grouped data (from fetch_data.py)
def load_grouped_data(file_path):
    with open(file_path, 'rb') as file:
        grouped_data = pickle.load(file)
    return grouped_data

# Convert grouped data into sentences
def convert_to_sentences(grouped_data):
    sentences = []
    for relationship, nodes in grouped_data.items():
        for start_node, end_node in nodes.items():
            sentence = f"{start_node} has a {relationship} with {end_node}."
            sentences.append(sentence)
    return sentences

# Chunk sentences into smaller parts
def chunk_sentences(sentences, chunk_size=10):
    return [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]

if __name__ == "__main__":
    # Load the grouped data
    grouped_data_path = "../output/grouped_data.pkl"  # Ensure this matches the fetch_data.py output
    grouped_data = load_grouped_data(grouped_data_path)

    # Convert to sentences
    sentences = convert_to_sentences(grouped_data)
    
    # Print the sentences before chunking
    print("================================ Sentences Before Chunking =================================")
    for i, sentence in enumerate(sentences):
        print(f"{i + 1}: {sentence}")
    print("===========================================================================================")

    # Chunk sentences
    chunks = chunk_sentences(sentences)

    # Save chunks to a file
    chunks_file_path = "../output/sentences_chunks.pkl"
    os.makedirs(os.path.dirname(chunks_file_path), exist_ok=True)
    with open(chunks_file_path, 'wb') as file:
        pickle.dump(chunks, file)

    print(f"Sentences and chunks saved to {chunks_file_path}")
