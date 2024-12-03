import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

# Pinecone credentials and index configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Check for required environment variables
if not PINECONE_API_KEY or not PINECONE_ENV or not PINECONE_INDEX:
    raise ValueError("PINECONE_API_KEY, PINECONE_ENV, and PINECONE_INDEX must be set in the .env file.")

# Initialize Pinecone client
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# Define index configuration
VECTOR_DIMENSION = 384  # Adjust based on your embedding model output
SIMILARITY_METRIC = "cosine"

def create_or_validate_index():
    """
    Creates the Pinecone index if it doesn't exist, or validates it if it does.
    """
    # Check if the index already exists
    existing_indexes = pinecone_client.list_indexes().names()
    
    if PINECONE_INDEX in existing_indexes:
        print(f"Index '{PINECONE_INDEX}' already exists. Validating configuration...")
        index_info = pinecone_client.describe_index(PINECONE_INDEX)
        print(f"Existing index configuration: {index_info}")

        # Validate index dimensions and similarity metric
        if index_info['dimension'] != VECTOR_DIMENSION:
            raise ValueError(f"Index dimension mismatch: expected {VECTOR_DIMENSION}, found {index_info['dimension']}")
        if index_info['metric'] != SIMILARITY_METRIC:
            raise ValueError(f"Index similarity metric mismatch: expected '{SIMILARITY_METRIC}', found '{index_info['metric']}'")

        print("Index configuration is valid.")
    else:
        print(f"Index '{PINECONE_INDEX}' does not exist. Creating index...")
        try:
            pinecone_client.create_index(
                name=PINECONE_INDEX,
                dimension=VECTOR_DIMENSION,
                metric=SIMILARITY_METRIC,
                spec=ServerlessSpec(
                    cloud="aws",  # Specify your cloud provider
                    region=PINECONE_ENV  # Use your Pinecone environment
                )
            )
            print(f"Index '{PINECONE_INDEX}' created successfully.")
        except Exception as e:
            raise RuntimeError(f"Error creating index: {e}")

if __name__ == "__main__":
    try:
        create_or_validate_index()
    except Exception as error:
        print(f"Error: {error}")
