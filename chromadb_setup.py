import chromadb
from core.embedding import MyEmbeddingFunction
import logging
import shutil
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ChromaDB setup
def setup_chromadb(path: str = "chroma_db", session_id: str = None):
    """
    Set up the ChromaDB client and create or retrieve a collection for storing document embeddings.
    Each session gets its own collection.

    Args:
        path (str, optional): The file path for the ChromaDB persistent storage. Default is "./chroma_db".
        session_id (str, optional): Unique identifier for the session. If provided, will create a session-specific collection.

    Returns:
        chromadb.api.Collection: The ChromaDB collection with embeddings ready for document storage and retrieval.
    """
    try:
        # Initialize the embedding function
        my_embed = MyEmbeddingFunction()
        
        # Verify embedding dimension by testing with a simple embedding
        test_embed = my_embed(["Test embedding initialization"])
        if test_embed and isinstance(test_embed, list) and len(test_embed) > 0:
            embedding_dim = len(test_embed[0])
            logger.info(f"Embedding function initialized with dimension: {embedding_dim}")
        else:
            logger.warning("Could not verify embedding dimension during initialization")

        # Initialize a persistent ChromaDB client
        chroma_client = chromadb.PersistentClient(path=path)

        # Create a collection name based on session_id if provided, otherwise use default
        collection_name = f"{session_id}" if session_id else "documents"

        # Create or retrieve the collection with custom embedding function
        collection = chroma_client.get_or_create_collection(
            name=collection_name, embedding_function=my_embed
        )

        logger.info(f"✅ ChromaDB setup complete. Collection '{collection_name}' is ready at: {path}")
        return collection
    except Exception as e:
        logger.error(f"Error setting up ChromaDB: {e}")
        raise

# Update the cleanup_chromadb function in chromadb_setup.py

def cleanup_chromadb(path: str = "./chroma_db", session_id: str = None):
    """
    Delete a session's collection or all collections.
    """
    try:
        if session_id:
            # Initialize a chromaDB client
            try:
                chroma_client = chromadb.PersistentClient(path=path)
                # Delete only the specific session's collection
                collection_name = f"{session_id}"
                try:
                    # First check if the collection exists
                    collections = chroma_client.list_collections()
                    collection_names = [collection.name for collection in collections]
                    
                    if collection_name in collection_names:
                        chroma_client.delete_collection(name=collection_name)
                        logger.info(f"Deleted collection '{collection_name}' for session: {session_id}")
                    else:
                        logger.warning(f"Collection '{collection_name}' not found for deletion")
                except Exception as e:
                    logger.error(f"Error deleting collection {collection_name}: {e}")
            except Exception as e:
                logger.error(f"Error connecting to ChromaDB: {e}")
        else:
            # On server restart, delete all collections by removing the DB directory
            if os.path.exists(path):
                try:
                    # List all collections first for logging
                    try:
                        chroma_client = chromadb.PersistentClient(path=path)
                        collections = chroma_client.list_collections()
                        collection_names = [collection.name for collection in collections]
                        logger.info(f"Found collections to delete: {collection_names}")
                        
                        # Delete each collection explicitly
                        for collection in collections:
                            try:
                                chroma_client.delete_collection(name=collection.name)
                                logger.info(f"Deleted collection: {collection.name}")
                            except Exception as e:
                                logger.error(f"Error deleting collection {collection.name}: {e}")
                    except Exception as e:
                        logger.error(f"Error listing collections: {e}")
                    
                    # Then remove the directory as a fallback
                    shutil.rmtree(path)
                    logger.info(f"Deleted all ChromaDB collections at {path}")
                except Exception as e:
                    logger.error(f"Error removing ChromaDB directory: {e}")
                # Recreate the directory
                os.makedirs(path, exist_ok=True)
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def list_collections(path: str = "./chroma_db"):
    """List all collections in the ChromaDB - compatible with all versions"""
    try:
        chroma_client = chromadb.PersistentClient(path=path)
        collections = chroma_client.list_collections()
        
        # Extract collection names
        collection_names = []
        for collection in collections:
            try:
                collection_names.append(collection.name)
            except AttributeError:
                # For older versions of ChromaDB
                if isinstance(collection, str):
                    collection_names.append(collection)
                else:
                    try:
                        collection_names.append(str(collection))
                    except:
                        pass
        
        logger.info(f"Found collections: {collection_names}")
        return collection_names
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        return []