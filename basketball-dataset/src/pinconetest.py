import os
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone
def init_pinecone(api_key, environment):
    try:
        pc = Pinecone(api_key=api_key)
        print("Successfully connected to Pinecone.")
        return pc
    except Exception as e:
        print("Failed to initialize Pinecone.")
        print(e)
        return None

# Create an index
def create_index(pc, index_name, dimension, region):
    try:
        # List existing indexes
        existing_indexes = pc.list_indexes().names()
        print(f"Existing indexes: {existing_indexes}")
        
        # Create the index if it doesn't exist
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=region
                )
            )
            print(f"Index '{index_name}' created successfully.")
        else:
            print(f"Index '{index_name}' already exists.")
    except Exception as e:
        print("An error occurred while creating the index:")
        print(e)

# Example usage
if __name__ == "__main__":
    API_KEY = "pcsk_6xctU4_UMcr3WqkYmE5K7eJsbcXVd6Q5AwUoVh3WtVFfb7NXRYUhWZqE4U3abD3cuUvvkT"
    ENVIRONMENT = "us-east-1"
    INDEX_NAME = "test-index"
    DIMENSION = 128

    # Initialize Pinecone
    pinecone_client = init_pinecone(api_key=API_KEY, environment=ENVIRONMENT)
    if pinecone_client:
        # Create index
        create_index(pinecone_client, INDEX_NAME, DIMENSION, region=ENVIRONMENT)
