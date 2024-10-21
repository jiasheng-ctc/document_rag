from flask import Flask, request, jsonify
import PyPDF2
import base64
from core.query import query_documents
from core.generate import (
    generate_CHITCHAT,
    generate_QUESTIONFROMDOC,
    generate_GENERALQUESTION,
    generate_SUMMARIZATION,
    generate_response,
    detect_task,
)
from core.document_utils import split_text
from core.embedding import ollama_embeddings
from chromadb_setup import setup_chromadb

app = Flask(__name__)

# Initialize the conversation history
CONVERSATION = []


def process_pdf(file):
    """Extract text from the uploaded PDF file."""
    text = ""
    pdf_reader = PyPDF2.PdfReader(file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text() + "\n"
    return text


def task(query: str, collection, pdf_files: bool) -> str:
    global CONVERSATION
    category = detect_task(str(CONVERSATION), query)
    respond = ""

    if "GENERAL QUESTION" in category:
        respond = generate_GENERALQUESTION(query, CONVERSATION)
    elif "SUMMARIZATION" in category:
        relevant_chunks = query_documents(collection, query)
        respond = generate_SUMMARIZATION(query, relevant_chunks, CONVERSATION)
    elif "QUESTION FROM DOCUMENTS" in category and pdf_files:
        relevant_chunks = query_documents(collection, query)
        respond = generate_QUESTIONFROMDOC(query, relevant_chunks, CONVERSATION)
    elif "CHIT CHAT" in category:
        respond = generate_CHITCHAT(query, CONVERSATION)
    else:
        relevant_chunks = query_documents(collection, query)
        respond = generate_response(query, relevant_chunks, CONVERSATION)

    return respond


@app.route("/ask", methods=["POST"])
def ask():
    """API endpoint to handle the user's query and PDF upload."""
    global CONVERSATION
    data = request.json
    question = data.get("question")
    pdf_content = data.get("pdf_content")

    if not question or not question.strip():
        return jsonify({"error": "Please enter a valid question."}), 400

    collection = setup_chromadb()

    # Process the uploaded PDF file and store the text in DOCUMENTS
    if pdf_content:
        try:
            pdf_data = base64.b64decode(pdf_content)
            doc = process_pdf(pdf_data)
            chunks = split_text(doc)
            for i, chunk in enumerate(chunks):
                embedding = ollama_embeddings(chunk)
                collection.upsert(
                    {
                        "id": f"uploaded_pdf_chunk{i+1}",
                        "text": chunk,
                        "embeddings": embedding,
                    }
                )
        except Exception as e:
            return jsonify({"error": f"Error processing PDF: {str(e)}"}), 400

    # Generate a response based on the question and documents
    response = task(question, collection, pdf_content is not None)

    # Append both the user question and the response to the conversation
    CONVERSATION.append({"role": "user", "content": question})
    CONVERSATION.append({"role": "assistant", "content": response})

    return jsonify(CONVERSATION)


if __name__ == "__main__":
    app.run(debug=True)
