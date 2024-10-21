import requests
import json


def ollama_chat(chat_history: str) -> str:
    """
    Call the Ollama model to generate a response for the given prompt.

    Args:
        chat_history: The history of chat between user and assistant.

    Returns:
        str: The generated response from the model, or an error message if an issue occurs.
    """
    url = "http://localhost:11434/api/chat"  # Ollama API endpoint for chat
    payload = {"model": "llama3", "messages": chat_history, "stream": False}

    try:
        # Send POST request
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an error for bad responses

        # Parse the JSON response
        json_response = response.json()

        # Extract the generated message content
        generated_text = json_response.get("message", {}).get(
            "content", "I couldn't understand. Could you explain more?"
        )
        return generated_text

    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return "An error occurred while generating the response."

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return "An error occurred while processing the response."


def ollama_generate(prompt: str) -> str:
    """
    Call the Ollama model to generate a response for the given prompt.

    Args:
        prompt (str): The input prompt from the user.

    Returns:
        str: The generated response from the model, or an error message if an issue occurs.
    """
    url = "http://localhost:11434/api/generate"  # Ollama API endpoint for chat
    payload = {"model": "llama3", "prompt": prompt, "stream": False}

    try:
        # Send POST request
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an error for bad responses

        # Parse the JSON response
        json_response = response.json()

        # Extract the generated message content
        generated_text = json_response.get("response", {})
        return generated_text

    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return "An error occurred while generating the response."

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return "An error occurred while processing the response."
