import requests
import json
from chromadb import Documents, EmbeddingFunction, Embeddings
from typing import Union, List, Dict, Any
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


DEFAULT_EMBEDDING_DIM = 4096

def ollama_embeddings(prompt: str, max_retries: int = 3, timeout: int = 300) -> Union[List[float], List[float]]:
    """
    Call the Ollama model to generate embeddings for the given prompt.
    Always returns a valid embedding (empty list in case of errors).

    Args:
        prompt (str): The input prompt for embedding generation.
        max_retries (int, optional): Maximum number of retries on failure. Default is 3.
        timeout (int, optional): Timeout for the API request in seconds. Default is 300.

    Returns:
        List[float]: The embedding as a list of floats. Returns an empty list in case of failure.
    """
    url = "http://localhost:11434/api/embeddings" 

    model_name = "llama3:70b"  

    if not prompt or not isinstance(prompt, str) or prompt.strip() == "":
        logger.warning("Empty or invalid prompt received. Returning empty embedding.")
        return []
    
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    
    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()  
            json_response = response.json()

            embeddings = json_response.get("embedding", [])
            if embeddings and isinstance(embeddings, list):
                logger.info(f"Successfully generated embedding with {len(embeddings)} dimensions using {model_name}")
                return embeddings
            else:
                logger.error("Invalid embedding format received from API")
                retries += 1
                if retries >= max_retries:
                    return []  

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API (attempt {retries+1}/{max_retries}): {e}")
            retries += 1
            if retries < max_retries:
                time.sleep(2 ** retries)
            else:
                logger.error(f"All retries failed: {e}")
                return [] 

        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error: {e}")
            retries += 1
            if retries >= max_retries:
                return []  

    return []

class MyEmbeddingFunction(EmbeddingFunction):
    """
    A custom embedding function that uses the Ollama model to generate embeddings for input documents.
    """
    
    def __init__(self, model_name: str = "llama3:70b", timeout: int = 60):
        """
        Initialize the embedding function with a model name and timeout.
        
        Args:
            model_name (str): Name of the Ollama model to use
            timeout (int): Timeout for API calls in seconds
        """
        self.model_name = model_name
        self.timeout = timeout
        self.url = "http://localhost:11434/api/embeddings"
        self.embedding_dim = None 

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
        if not isinstance(input, list):
            input = [input]
        
        all_embeddings = []

        for document in input:
            if not document or not isinstance(document, str):
                logger.warning("Empty or non-string document received. Using zero vector.")
                all_embeddings.append([0.0] * self.embedding_dim)
                continue
                
            payload = {
                "model": self.model_name,
                "prompt": document,
                "stream": False
            }
            
            try:
                response = requests.post(self.url, json=payload, timeout=self.timeout)
                response.raise_for_status()

                json_response = response.json()
                embedding = json_response.get("embedding", [])

                if not embedding or not isinstance(embedding, list):
                    logger.error(f"Invalid embedding received: {embedding}")
                    all_embeddings.append([0.0] * self.embedding_dim)
                else:
                    if self.embedding_dim is None or self.embedding_dim != len(embedding):
                        self.embedding_dim = len(embedding)
                        logger.info(f"Updated embedding dimension to {self.embedding_dim}")
                        
                    all_embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                all_embeddings.append([0.0] * self.embedding_dim)
        
        return all_embeddings