import requests
import sys
import logging
from chromadb_setup import setup_chromadb
from core.embedding import ollama_embeddings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ollama_connection():
    """Test connection to Ollama server"""
    try:
        logger.info("Testing connection to Ollama server...")
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        
        if not models:
            logger.warning("Connected to Ollama, but no models found!")
            return False
            
        logger.info(f"Successfully connected to Ollama. Available models: {[m['name'] for m in models]}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        return False

def test_embedding():
    """Test the embedding functionality"""
    try:
        logger.info("Testing embedding generation...")
        embedding = ollama_embeddings("This is a test query")
        
        if isinstance(embedding, list) and len(embedding) > 0:
            logger.info(f"Successfully generated embedding with {len(embedding)} dimensions")
            return True
        else:
            logger.warning(f"Embedding generation returned unexpected result: {embedding}")
            return False
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        return False

def test_embedding_dimensions():
    """Test the embedding dimensions to ensure consistency"""
    try:
        logger.info("Testing embedding dimensions...")
        
        # Test with different text inputs
        texts = [
            "This is a short test",
            "This is a longer test with more words to see if length affects embedding",
            "Another test to verify embedding dimensions"
        ]
        
        dimensions = set()
        for text in texts:
            embedding = ollama_embeddings(text)
            if isinstance(embedding, list):
                dim = len(embedding)
                dimensions.add(dim)
                logger.info(f"Generated embedding with {dim} dimensions")
            else:
                logger.warning(f"Failed to generate embedding: {embedding}")
        
        if len(dimensions) == 1:
            logger.info(f"All embeddings have consistent dimension: {next(iter(dimensions))}")
            return True
        else:
            logger.error(f"Inconsistent embedding dimensions detected: {dimensions}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to test embedding dimensions: {e}")
        return False

def test_chromadb():
    """Test ChromaDB setup and basic functionality"""
    try:
        logger.info("Testing ChromaDB setup...")
        collection = setup_chromadb()
        
        # Try adding a test document
        test_text = "This is a test document for ChromaDB"
        embedding = ollama_embeddings(test_text)
        
        if isinstance(embedding, list):
            collection.upsert(
                ids=["test_document"],
                documents=[test_text],
                embeddings=[embedding]
            )
            
            # Try querying it back
            results = collection.query(
                query_texts=["test"],
                n_results=1
            )
            
            if results and len(results["documents"]) > 0:
                logger.info("Successfully added and retrieved document from ChromaDB")
                return True
            else:
                logger.warning("Could not retrieve document from ChromaDB")
                return False
        else:
            logger.warning("Could not generate embedding for ChromaDB test")
            return False
    except Exception as e:
        logger.error(f"Failed to set up or use ChromaDB: {e}")
        return False

if __name__ == "__main__":
    success = True
    
    # Test Ollama connection
    if not test_ollama_connection():
        success = False
        
    # Test embedding
    if not test_embedding():
        success = False
    
    # Test embedding dimensions
    if not test_embedding_dimensions():
        success = False
    
    # Test ChromaDB
    if not test_chromadb():
        success = False
    
    if success:
        logger.info("All connection tests passed! You can now start the FastAPI application.")
        sys.exit(0)
    else:
        logger.error("One or more connection tests failed. Please check the logs above.")
        sys.exit(1)