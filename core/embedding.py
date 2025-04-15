import requests
import json
from chromadb import Documents, EmbeddingFunction, Embeddings
from typing import Union, List, Dict, Any
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the default embedding dimension for LLaMA models
# LLaMA 3's embedding dimension is 4096, but we'll check this dynamically
DEFAULT_EMBEDDING_DIM = 4096

# Update in ollama_embeddings function
def ollama_embeddings(prompt: str, max_retries: int = 3, timeout: int = 300) -> Union[List[float], str]:
    """
    Call the Ollama model to generate embeddings for the given prompt.

    Args:
        prompt (str): The input prompt for embedding generation.
        max_retries (int, optional): Maximum number of retries on failure. Default is 3.
        timeout (int, optional): Timeout for the API request in seconds. Default is 300.

    Returns:
        Union[List[float], str]: The embedding as a list of floats, or an error message if an issue occurs.
    """
    url = "http://localhost:11434/api/embeddings"  # Ollama API endpoint for embeddings
    
    # Use the same model as in the llama.py file for consistency
    model_name = "phi"  # This should match the model in llama.py
    
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    
    retries = 0
    while retries < max_retries:
        try:
            # Send POST request with timeout
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()  # Raise an error for bad responses

            # Parse the JSON response
            json_response = response.json()

            # Extract the embeddings
            embeddings = json_response.get("embedding", [])
            if embeddings and isinstance(embeddings, list):
                logger.info(f"Successfully generated embedding with {len(embeddings)} dimensions using {model_name}")
            return embeddings

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API (attempt {retries+1}/{max_retries}): {e}")
            retries += 1
            if retries < max_retries:
                # Exponential backoff
                time.sleep(2 ** retries)
                continue
            return f"An error occurred while generating the embeddings: {str(e)}"

        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error: {e}")
            return f"An error occurred while processing the embeddings response: {str(e)}"

class MyEmbeddingFunction(EmbeddingFunction):
    """
    A custom embedding function that uses the Ollama model to generate embeddings for input documents.
    """
    
    def __init__(self, model_name: str = "phi", timeout: int = 60):
        """
        Initialize the embedding function with a model name and timeout.
        
        Args:
            model_name (str): Name of the Ollama model to use
            timeout (int): Timeout for API calls in seconds
        """
        self.model_name = model_name
        self.timeout = timeout
        self.url = "http://localhost:11434/api/embeddings"
        self.embedding_dim = None  # Will be set dynamically on first call
        
        # Try to determine embedding dimension at initialization
        self._initialize_embedding_dim()
    
    def _initialize_embedding_dim(self):
        """Get the embedding dimension by generating a test embedding"""
        try:
            test_embedding = ollama_embeddings("Test embedding dimension")
            if isinstance(test_embedding, list):
                self.embedding_dim = len(test_embedding)
                logger.info(f"Detected LLaMA embedding dimension: {self.embedding_dim}")
            else:
                logger.warning("Could not determine embedding dimension, using default")
                self.embedding_dim = DEFAULT_EMBEDDING_DIM
        except Exception as e:
            logger.warning(f"Error detecting embedding dimension: {e}. Using default: {DEFAULT_EMBEDDING_DIM}")
            self.embedding_dim = DEFAULT_EMBEDDING_DIM

    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for a list of input documents by calling the Ollama API.

        Args:
            input (Documents): A list of documents (strings) for which to generate embeddings.

        Returns:
            Embeddings: A list of embedding vectors corresponding to the input documents.
        """
        # Ensure input is a list
        if not isinstance(input, list):
            input = [input]
        
        all_embeddings = []
        
        # Process each document separately to get individual embeddings
        for document in input:
            if not document or not isinstance(document, str):
                # Handle empty or non-string documents with a zero vector
                logger.warning("Empty or non-string document received. Using zero vector.")
                all_embeddings.append([0.0] * self.embedding_dim)
                continue
                
            payload = {
                "model": self.model_name,
                "prompt": document,
                "stream": False
            }
            
            try:
                # Send POST request to Ollama API with timeout
                response = requests.post(self.url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                
                # Parse the JSON response to get embeddings
                json_response = response.json()
                embedding = json_response.get("embedding", [])
                
                # Verify embedding is valid
                if not embedding or not isinstance(embedding, list):
                    logger.error(f"Invalid embedding received: {embedding}")
                    all_embeddings.append([0.0] * self.embedding_dim)
                else:
                    # If this is our first successful embedding, update the dimension
                    if self.embedding_dim is None or self.embedding_dim != len(embedding):
                        self.embedding_dim = len(embedding)
                        logger.info(f"Updated embedding dimension to {self.embedding_dim}")
                        
                    all_embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                # Return a zero vector as a fallback
                all_embeddings.append([0.0] * self.embedding_dim)
        
        return all_embeddings