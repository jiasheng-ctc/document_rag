import chromadb
from core.embedding import MyEmbeddingFunction


# ChromaDB setup
def setup_chromadb(path: str = "chroma_db") -> chromadb.api.Collection:
    """
    Set up the ChromaDB client and create or retrieve a collection for storing document embeddings.

    Args:
        path (str, optional): The file path for the ChromaDB persistent storage. Default is "chroma_db".

    Returns:
        chromadb.api.Collection: The ChromaDB collection with embeddings ready for document storage and retrieval.
    """
    # Initialize the embedding function
    my_embed = MyEmbeddingFunction()

    # Initialize a persistent ChromaDB client
    chroma_client = chromadb.PersistentClient(path=path)

    # Create or retrieve the "documents" collection with custom embedding function
    collection = chroma_client.get_or_create_collection(
        name="documents", embedding_function=my_embed
    )

    print(f"✅ ChromaDB setup complete. Collection 'documents' is ready at: {path}")

    return collection
