import requests
import json
from chromadb import Documents, EmbeddingFunction, Embeddings
from typing import Union, List


def ollama_embeddings(prompt: str) -> Union[List[float], str]:
    """
    Call the Ollama model to generate embeddings for the given prompt.

    Args:
        prompt (str): The input prompt for embedding generation.

    Returns:
        Union[List[float], str]: The embedding as a list of floats, or an error message if an issue occurs.
    """
    url = "http://localhost:11434/api/embeddings"  # Ollama API endpoint for embeddings
    payload = {"model": "llama3", "prompt": prompt, "stream": False}

    try:
        # Send POST request
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an error for bad responses

        # Parse the JSON response
        json_response = response.json()

        # Extract the embeddings
        embeddings = json_response.get("embedding", [])
        return embeddings

    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return "An error occurred while generating the embeddings."

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return "An error occurred while processing the embeddings response."


class MyEmbeddingFunction(EmbeddingFunction):
    """
    A custom embedding function that uses the Ollama model to generate embeddings for input documents.
    """

    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for a list of input documents by calling the Ollama API.

        Args:
            input (Documents): A list of documents (strings) for which to generate embeddings.

        Returns:
            Embeddings: A list of embedding vectors corresponding to the input documents.
        """
        # Combine multiple input documents into a single prompt
        prompt = "\n".join(input)  # Ensure input is a list of strings

        url = "http://localhost:11434/api/embeddings"  # Ollama API endpoint for embeddings
        payload = {"model": "llama3", "prompt": prompt, "stream": False}

        try:
            # Send POST request to Ollama API
            response = requests.post(url, json=payload)
            response.raise_for_status()  # Raise an error for bad responses

            # Parse the JSON response to get embeddings
            json_response = response.json()
            embeddings = json_response.get("embedding", [])
            return embeddings

        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return (
                []
            )  # Return an empty list on error to maintain the expected return type

        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            return []  # Return an empty list on JSON parsing error
