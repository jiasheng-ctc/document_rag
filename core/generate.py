from core.llama import ollama_chat, ollama_generate
import logging
import re

logger = logging.getLogger(__name__)

def clean_chunk_text(text):
    """
    Clean up text chunks to improve readability for the LLM.
    This helps the model better understand the content.
    """
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\$\s*(\d+)\s*\.\s*(\d+)', r'$\1.\2', text)

    text = text.replace('S G G S T', 'SG GST')  
    text = text.replace('G S T', 'GST') 
    text = re.sub(r'\s[bcdefghijklmnopqrstuvwxyz]\s', ' ', text, flags=re.IGNORECASE)
    
    return text.strip()

def detect_task(conversation: str, query: str) -> str:
    """Detect the type of task from the conversation and query."""
    if len(conversation) > 1000:
        conversation = conversation[-1000:]
    
    prompt = (
        "You are an AI assistant tasked with analyzing the following conversation between a user and an assistant. "
        "Your goal is to determine if the user is asking for a SUMMARIZATION of a document or asking a QUESTION FROM DOCUMENTS. "
        "Choose one of these two categories that best describes the user's request: "
        "'SUMMARIZATION' or 'QUESTION FROM DOCUMENTS'."
        "\n\nHere is the conversation:\n\n" + conversation + "\n\n"
        "User's final message: " + query + "\n\n"
        "Respond only with the category name: either 'SUMMARIZATION' or 'QUESTION FROM DOCUMENTS'."
    )

    category = ollama_generate(prompt)
    logger.info(f"Detected task category: {category}")
    return category

def generate_SUMMARIZATION(query: str, relevant_chunks: list, history: list) -> str:
    """Generate a summary based on relevant document chunks."""

    cleaned_chunks = [clean_chunk_text(chunk) for chunk in relevant_chunks]
    context = "\n\n".join(cleaned_chunks)
    if len(context) > 1500:
        context = context[:1500] + "... [content truncated]"

    prompt = (
        "You are a precise document summarization assistant. The user has a document that may have extraction artifacts "
        "like extra spaces between characters, especially in numbers (for example '$ 1 0 . 9 0' should be read as '$10.90'). "
        "Your task is to summarize ONLY the information provided in the document chunks below. "
        "Do NOT include any information that is not explicitly present in the document chunks. "
        "If the document chunks do not contain information related to the user's request, state this clearly. "
        "Never make up or infer information that isn't directly stated in the documents."
        "\n\nUser's request: " + query + 
        "\n\nDocument content to summarize (ONLY use this information):\n" + context + 
        "\n\nSummary (based STRICTLY on the provided document content):"
    )

    history = history[-3:] if len(history) > 3 else history.copy()
    history.append({"role": "user", "content": prompt})

    response = ollama_chat(history)
    return response

def generate_QUESTIONFROMDOC(query: str, relevant_chunks: list, history: list) -> str:
    """Answer a question based on document context with enhanced instructions for handling poorly formatted text."""

    cleaned_chunks = [clean_chunk_text(chunk) for chunk in relevant_chunks]
    context = "\n\n".join(cleaned_chunks)
    if len(context) > 1500:
        context = context[:1500] + "... [content truncated]"

    prompt = (
        "You are a precise document question-answering assistant. The user has a question about information "
        "in a PDF document. The text from this PDF may have extraction artifacts like extra spaces between characters "
        "or unusual formatting. Your task is to answer the user's question using ONLY the information "
        "provided in the document chunks below. Follow these rules: "
        
        "1. If the text contains oddly spaced characters like '$ 1 0 . 9 0', interpret this as the properly formatted value '$10.90'. "
        "2. Fix any obvious formatting issues when interpreting the content. "
        "3. If the answer is not contained in the documents, explicitly state: 'I cannot find this information in the provided documents.' "
        "4. NEVER make up information not found in these documents. "
        "5. Be concise and direct in your answer. "
        "6. You do not need to tell information can be found in which document chunks. "
        "\n\nQuestion: " + query + 
        "\n\nDocument content (this may contain PDF extraction artifacts like extra spaces between characters):\n" + context + 
        "\n\nAnswer (interpret any formatting issues and answer based STRICTLY on the provided content):"
    )

    history = history[-3:] if len(history) > 3 else history.copy()
    history.append({"role": "user", "content": prompt})

    response = ollama_chat(history)
    return response