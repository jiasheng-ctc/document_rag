from core.web_scrape import scrape_web
from core.llama import ollama_chat, ollama_generate


def detect_task(conversation: str, query: str) -> str:
    prompt = (
        "You are an AI assistant tasked with analyzing the following conversation between a user and an assistant. "
        "Your goal is to determine the specific task the user is requesting, particularly focusing on the user's final message, "
        "as the user may shift the conversation and request something entirely new. "
        "Carefully analyze the final message to determine the nature of the task being requested."
        "\n\nChoose one of the following categories that best describes the user's final request: "
        "'GENERAL QUESTION', 'SUMMARIZATION', 'QUESTION FROM DOCUMENTS', 'CHIT CHAT', or 'OTHER'."
        "\n\nHere is the entire conversation:\n\n" + conversation + "\n\n"
        "User's final message: " + query + "\n\n"
        "Respond only with the category that most accurately matches the user's final request."
    )

    category = ollama_generate(prompt)
    return category


def generate_GENERALQUESTION(query: str, history: list) -> str:
    # Scrape the web for relevant information
    web_result = scrape_web(query)

    # Construct a precise prompt
    prompt = (
        "You are an expert assistant answering questions based on web search results. "
        "Use only relevant information from the web search to answer the user's query. "
        "Ignore any irrelevant details, and provide a concise response in up to three sentences if the query doesn't specify otherwise."
        "Don't mention anything about the resources when you got your respond"
        "\n\nQuestion: " + query + "\n\nWeb Result:\n" + web_result + "\n\nAnswer:"
    )
    history = history.copy()
    history.append({"role": "user", "content": prompt})

    # Generate response using the Ollama model
    response = ollama_chat(history)
    return response


def generate_SUMMARIZATION(query: str, relevant_chunks: str, history: list) -> str:
    # Combine relevant chunks into a single context
    context = "\n\n".join(relevant_chunks)

    # Refined prompt for summarization
    prompt = (
        "You are an AI assistant tasked with summarizing information. "
        "If the user's query provides context, summarize based on the query. "
        "If the query lacks context, use the provided relevant chunks to generate the summary."
        "Don't mention anything about the resources when you got your respond"
        "\n\nQuery: " + query + "\n\nRelevant Chunks:\n" + context + "\n\nSummary:"
    )

    history = history.copy()
    history.append({"role": "user", "content": prompt})

    # Generate response using the Ollama model
    response = ollama_chat(history)
    return response


def generate_QUESTIONFROMDOC(query: str, relevant_chunks: str, history: list) -> str:
    # Combine relevant chunks into a single context
    context = "\n\n".join(relevant_chunks)

    # Refined prompt to ensure focus on document context
    prompt = (
        "You are an AI assistant answering questions based on provided document context. "
        "Only use the relevant information from the document to answer the user's query. "
        "Ignore any irrelevant details, and provide a concise response in up to three sentences."
        "Don't mention anything about the resources when you got your respond"
        "\n\nQuestion: " + query + "\n\nRelevant Context:\n" + context + "\n\nAnswer:"
    )

    history = history.copy()
    history.append({"role": "user", "content": prompt})

    # Generate response using the Ollama model
    response = ollama_chat(history)
    return response


def generate_CHITCHAT(query: str, history: list) -> str:
    # Use the user query directly for chit-chat
    prompt = query

    history = history.copy()
    history.append({"role": "user", "content": prompt})

    # Generate chit-chat response using the Ollama model
    response = ollama_chat(history)
    return response


def generate_response(question: str, relevant_chunks: list, history: list) -> str:
    """
    Generate a response to the given question based on retrieved context and web scraping results.

    Args:
        question (str): The question to answer.
        relevant_chunks (list): A list of relevant text chunks retrieved from documents.

    Returns:
        str: The generated answer to the question, based on the provided context and web results.
    """
    # Combine relevant chunks into a single context
    context = "\n\n".join(relevant_chunks)

    # Scrape the web for additional relevant information
    web_result = scrape_web(question)

    # Refined prompt to combine both sources effectively
    prompt = (
        "You are an AI assistant designed for question-answering tasks. "
        "Use both the provided context from documents and the web search results to answer the user's query. "
        "If any part of the context or web result seems irrelevant to the question, ignore it."
        "Don't mention anything about the resources when you got your respond"
        "\n\nDocument Context:\n"
        + context
        + "\n\nWeb Result:\n"
        + web_result
        + "\n\nQuestion:\n"
        + question
        + "\n\nAnswer (in three concise sentences):"
    )

    history = history.copy()
    history.append({"role": "user", "content": prompt})

    # Generate the response using the Ollama model
    response = ollama_chat(history)
    return response
