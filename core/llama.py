import requests
import json
import logging
import time
from typing import List, Dict, Any, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ollama_chat(chat_history: str, model_name: str = "phi", max_retries: int = 2, timeout: int = 120) -> str:
    """
    Call the Ollama model to generate a response for the given prompt with retry logic.

    Args:
        chat_history: The history of chat between user and assistant.
        model_name: The name of the Ollama model to use.
        max_retries: Maximum number of retry attempts.
        timeout: Timeout for the API call in seconds.

    Returns:
        str: The generated response from the model, or an error message if an issue occurs.
    """
    url = "http://localhost:11434/api/chat"  # Ollama API endpoint for chat
    
    # Ensure chat history isn't too long
    if isinstance(chat_history, list) and len(chat_history) > 10:
        chat_history = chat_history[-10:]  # Keep only the last 10 messages
    
    payload = {"model": model_name, "messages": chat_history, "stream": False}
    
    # Add some model parameters to potentially speed up response
    payload["options"] = {
        "temperature": 0.1,       # Reduced from 0.7 to 0.1 for more deterministic outputs
        "top_p": 0.5,             # Reduced from 0.9 to 0.5 to restrict token sampling
        "num_predict": 512,       # Increased to allow for complete answers
        "top_k": 10,              # Added to restrict to only the most likely tokens
        "repeat_penalty": 1.2     # Added to discourage repetition which can indicate confabulation
    }

    retries = 0
    while retries <= max_retries:
        try:
            logger.info(f"Sending chat request to Ollama with model: {model_name} (attempt {retries+1}/{max_retries+1})")
            # Send POST request
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()  # Raise an error for bad responses

            # Parse the JSON response
            json_response = response.json()

            # Extract the generated message content
            generated_text = json_response.get("message", {}).get(
                "content", "I couldn't understand. Could you explain more?"
            )
            return generated_text

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout error on attempt {retries+1}/{max_retries+1}. Retrying...")
            retries += 1
            # If this was the last retry, give up
            if retries > max_retries:
                return "I'm sorry, but I'm having trouble generating a response at the moment due to system load. Please try again with a simpler question or try again later."
            # Wait before retrying
            time.sleep(2)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            retries += 1
            if retries > max_retries:
                return f"An error occurred while generating the response: {str(e)}"
            time.sleep(2)

        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error: {e}")
            return f"An error occurred while processing the response: {str(e)}"

def ollama_generate(prompt: str, model_name: str = "phi", timeout: int = 60) -> str:
    """
    Call the Ollama model to generate a response for the given prompt.

    Args:
        prompt (str): The input prompt from the user.
        model_name: The name of the Ollama model to use.
        timeout: Timeout for the API call in seconds.

    Returns:
        str: The generated response from the model, or an error message if an issue occurs.
    """
    url = "http://localhost:11434/api/generate"  # Ollama API endpoint for generate
    
    # Limit prompt size if it's very large
    if len(prompt) > 4000:
        prompt = prompt[:4000]
        
    payload = {
        "model": model_name, 
        "prompt": prompt, 
        "stream": False,
        "options": {
            "temperature": 0.1,       # Reduced from 0.7 to match chat parameters
            "top_p": 0.5,             # Reduced to match chat parameters
            "top_k": 10,              # Added to restrict token selection
            "num_predict": 64,        # Task detection doesn't need many tokens
            "repeat_penalty": 1.2     # Added to discourage repetition
        }
    }

    try:
        logger.info(f"Sending generate request to Ollama with model: {model_name}")
        # Send POST request
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()  # Raise an error for bad responses

        # Parse the JSON response
        json_response = response.json()

        # Extract the generated message content
        generated_text = json_response.get("response", "")
        return generated_text

    except requests.exceptions.Timeout:
        logger.error(f"Timeout error when calling Ollama generate API")
        return "QUESTION FROM DOCUMENTS"  # Default to question answering on timeout

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Ollama API: {e}")
        return "QUESTION FROM DOCUMENTS"  # Default to question answering on error

    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error: {e}")
        return "QUESTION FROM DOCUMENTS"  # Default to question answering on parse error