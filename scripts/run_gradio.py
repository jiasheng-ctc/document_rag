import gradio as gr
import PyPDF2
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


def handle_question_and_documents(question: str, pdf_file) -> list:
    """Handle the user's query and process the uploaded PDF."""
    if not question.strip():
        return [{"role": "user", "content": "Please enter a valid question."}]

    collection = setup_chromadb()

    # Process the uploaded PDF file and store the text in DOCUMENTS
    if pdf_file:
        try:
            doc = process_pdf(pdf_file)
            chunks = split_text(doc)
            for i, chunk in enumerate(chunks):
                embedding = ollama_embeddings(chunk)
                collection.upsert(
                    {
                        "id": f"{pdf_file.name}_chunk{i+1}",
                        "text": chunk,
                        "embeddings": embedding,
                    }
                )
        except Exception as e:
            return [
                {
                    "role": "user",
                    "content": f"Error processing {pdf_file.name}: {str(e)}",
                }
            ]

    # Generate a response based on the question and documents
    response = task(question, collection, pdf_file is not None)

    # Append both the user question and the response to the conversation
    CONVERSATION.append({"role": "user", "content": question})
    CONVERSATION.append({"role": "assistant", "content": response})

    return CONVERSATION  # Return the entire conversation history


# Create the Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# LLaMA-based RAG System (AI Assistant)")
    gr.Markdown(
        "## Ask Your Question and Upload a PDF for Smart Answers or Summaries\nGet detailed answers or concise summaries based on your query and the content of your uploaded PDF. Perfect for document-based question answering, summarization, and more!"
    )

    chatbot = gr.Chatbot(label="Conversation", type="messages")  # Chat interface
    with gr.Row():
        question_input = gr.Textbox(
            label="Ask a question:",
            elem_id="question",
            placeholder="Type your question here...",
            lines=1,
        )  # Question input

        # Smaller styled upload button
        pdf_upload = gr.UploadButton(
            label="📎", elem_id="attach-button", file_types=[".pdf"], interactive=True
        )  # PDF upload
        submit_button = gr.Button("Send", elem_id="submit")  # Submit button

    # Set the action for the submit button
    submit_button.click(
        handle_question_and_documents,
        inputs=[question_input, pdf_upload],
        outputs=chatbot,
    )

iface.css = """
#attach-button{
    width: 10px;  /* Button width */
    height: 90px;  /* Button height */
    font-size: 24px;  /* Adjust font size for the icon */
    background-color: #007bff;  /* Blue background */
    color: white;
    border: none;
    cursor: pointer;  /* Pointer cursor on hover */
}
#question{
    width: 200px;
    flex-grow: 1;
}
#submit{
    width: 20px;
    height: 90px;
    background-color: #2E8B57;
}
"""

# Launch the app
if __name__ == "__main__":
    iface.launch()
