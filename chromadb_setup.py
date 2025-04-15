import chromadb
from core.embedding import MyEmbeddingFunction
import logging
import shutil
import os
import time

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

def cleanup_chromadb(path: str = "./chroma_db", session_id: str = None):
    """
    Delete a session's collection or all collections.
    """
    try:
        if session_id:
            # Initialize a chromaDB client
            try:
                chroma_client = chromadb.PersistentClient(path=path)
                
                # Get all collections
                all_collections = chroma_client.list_collections()
                
                # Extract collection names consistently
                collection_names = [extract_collection_name(coll) for coll in all_collections]
                
                if session_id in collection_names:
                    logger.info(f"Found collection '{session_id}' - attempting to delete")
                    
                    # Add a small delay to ensure any pending operations complete
                    time.sleep(0.5)
                    
                    # Delete the collection
                    chroma_client.delete_collection(name=session_id)
                    
                    # Verify deletion
                    collections_after = chroma_client.list_collections()
                    collection_names_after = [extract_collection_name(coll) for coll in collections_after]
                    
                    if session_id not in collection_names_after:
                        logger.info(f"Successfully deleted collection '{session_id}'")
                        return True
                    else:
                        logger.warning(f"Collection '{session_id}' still exists after deletion attempt")
                        return False
                else:
                    logger.warning(f"Collection '{session_id}' not found for deletion")
                    return False
                    
            except Exception as e:
                logger.error(f"Error when deleting collection {session_id}: {e}")
                return False
        else:
            # On server restart, delete all collections
            if os.path.exists(path):
                try:
                    # List all collections first for logging
                    try:
                        chroma_client = chromadb.PersistentClient(path=path)
                        collections = chroma_client.list_collections()
                        
                        # Extract collection names consistently
                        collection_names = [extract_collection_name(coll) for coll in collections]
                        logger.info(f"Found collections to delete: {collection_names}")
                        
                        # Delete each collection explicitly
                        success = True
                        for coll_name in collection_names:
                            try:
                                # Add a small delay between deletions
                                time.sleep(0.5)
                                chroma_client.delete_collection(name=coll_name)
                                logger.info(f"Deleted collection: {coll_name}")
                            except Exception as e:
                                logger.error(f"Error deleting collection {coll_name}: {e}")
                                success = False
                        
                        # Verify all collections were deleted
                        remaining = chroma_client.list_collections()
                        if remaining and not success:
                            # If we couldn't delete all collections properly, try directory removal
                            logger.warning("Some collections couldn't be deleted. Trying directory removal.")
                            shutil.rmtree(path)
                            logger.info(f"Deleted ChromaDB directory at {path}")
                            # Recreate the directory
                            os.makedirs(path, exist_ok=True)
                        
                    except Exception as e:
                        logger.error(f"Error listing collections: {e}")
                        # Fall back to deleting the directory
                        shutil.rmtree(path)
                        logger.info(f"Deleted all ChromaDB collections at {path}")
                        # Recreate the directory
                        os.makedirs(path, exist_ok=True)
                except Exception as e:
                    logger.error(f"Error removing ChromaDB directory: {e}")
                    return False
            return True
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return False

def extract_collection_name(collection):
    """
    Extract the name from a collection object in a standardized way.
    Works with different ChromaDB versions that might return collections
    in different formats.
    
    Args:
        collection: A collection object from ChromaDB, which could be an object,
                   dictionary, or string depending on the ChromaDB version.
    
    Returns:
        str: The name of the collection, or a string representation if name cannot be extracted.
    """
    try:
        # Try to access name as an attribute (newer ChromaDB versions)
        if hasattr(collection, 'name'):
            return collection.name
        # Try to access name as a dictionary key
        elif isinstance(collection, dict) and 'name' in collection:
            return collection['name']
        # If it's already a string, return it directly
        elif isinstance(collection, str):
            return collection
        # For any other case, convert to string
        else:
            return str(collection)
    except Exception:
        # If all fails, return a safe string representation
        return str(collection)
        
def list_collections(path: str = "./chroma_db"):
    """List all collections in the ChromaDB - compatible with all versions"""
    try:
        chroma_client = chromadb.PersistentClient(path=path)
        collections = chroma_client.list_collections()
        
        # Extract collection names
        collection_names = []
        for collection in collections:
            try:
                if hasattr(collection, 'name'):
                    collection_names.append(collection.name)
                else:
                    # For older versions of ChromaDB
                    if isinstance(collection, str):
                        collection_names.append(collection)
                    else:
                        collection_names.append(str(collection))
            except Exception as e:
                logger.error(f"Error getting collection name: {e}")
                continue
        
        logger.info(f"Found collections: {collection_names}")
        return collections
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        return []