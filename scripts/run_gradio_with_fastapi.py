import gradio as gr
import PyPDF2
import base64
import requests
import json
import uuid
import os
import time
import atexit

# FastAPI server URL
FASTAPI_URL = "http://localhost:8000"  # Change this if your FastAPI server is on a different host/port

# Initialize the session ID and document tracking
SESSION_ID = str(uuid.uuid4())
UPLOADED_PDFS = []

# Get the current directory to properly reference the logo
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to find the root directory (assuming scripts is one level below root)
root_dir = os.path.dirname(current_dir)
logo_path = os.path.join(root_dir, "public", "logo.svg")

print(f"Working directory: {os.getcwd()}")
print(f"Logo path: {logo_path}")

# Try to read the logo file directly
logo_base64 = None
if os.path.exists(logo_path):
    try:
        with open(logo_path, 'rb') as f:
            logo_data = f.read()
            # Convert to base64 for embedding directly in HTML
            logo_base64 = base64.b64encode(logo_data).decode('utf-8')
            print(f"Successfully loaded logo: {len(logo_data)} bytes")
    except Exception as e:
        print(f"Error loading logo: {e}")

# Custom CSS to match the Synapxe style
custom_css = """
:root {
    --primary-color: #000000;
    --secondary-color: #000000;
    --background-color: #FFFFFF;
    --text-color: #000000;
    --light-gray: #F5F5F5;
    --border-color: #E0E0E0;
}

body {
    font-family: 'Arial', sans-serif;
}

.logo-container {
    padding: 20px;
    background-color: var(--background-color);
    border-bottom: 1px solid var(--border-color);
    display: flex;
    align-items: center;
    margin-bottom: 20px;
}

.logo-container img {
    height: 60px;
}

.logo-container .tagline {
    margin-left: 15px;
    color: var(--primary-color);
    font-size: 1.2em;
    font-weight: bold;
}

.chatbot-container {
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}


.action-btn {
    background-color: transparent !important;
    color: #000000 !important;  /* Black text */
    border: 1px solid var(--border-color) !important;
    padding: 8px 15px !important;
    border-radius: 20px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    font-weight: normal !important;
}

.action-btn:hover {
    background-color: rgba(0, 0, 0, 0.05) !important;
    border-color: #000000 !important;
}

.status-indicator {
    padding: 10px 15px;
    border-radius: 5px;
    background-color: var(--light-gray);
    font-size: 0.9em;
}

#question textarea {
    border-radius: 20px !important;
    border: 1px solid var(--border-color) !important;
    padding: 12px 15px !important;
}

#submit {
    background-color: transparent !important;
    color: #000000 !important;  /* Black text */
    border: 1px solid var(--border-color) !important;
    border-radius: 20px !important;
    transition: all 0.2s ease !important;
}

#submit:hover {
    background-color: rgba(0, 0, 0, 0.05) !important;
    border-color: #000000 !important;
}

body, p, h1, h2, h3, button, input, textarea {
    color: #000000 !important;
}

footer p {
    color: #555555 !important;  /* Slightly lighter for footer */
}

/* Button styling - works in older Gradio versions */
button.primary {
    background-color: var(--primary-color) !important;
    color: white !important;
}

.upload-section {
    background-color: var(--light-gray);
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 15px;
}

/* Logo placeholder styling */
.logo-placeholder {
    background-color: #f0f0f0 !important;
    color: #000000 !important;
}
"""

# Functions remain the same
def debug_collections():
    """Debug function to check existing collections."""
    try:
        response = requests.get(f"{FASTAPI_URL}/list-collections")
        if response.status_code == 200:
            collections = response.json().get("collections", [])
            print(f"Server collections: {collections}")
            
            # Check if our session exists
            session_collection = f"{SESSION_ID}"
            if collections and session_collection in collections:
                print(f"✅ Current session collection '{session_collection}' exists on server")
            else:
                print(f"❌ Current session collection '{session_collection}' NOT FOUND on server")
            
            return collections
        else:
            print(f"Error response: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Error checking collections: {e}")
        return []

def process_pdfs(files):
    """Extract text from the uploaded PDF files and convert to base64."""
    global UPLOADED_PDFS
    
    if not files:
        return None, "No documents uploaded"
    
    processed_files = []
    for file in files:
        try:
            # For UploadButton, file is a path to the temporary file
            with open(file.name, 'rb') as f:
                pdf_bytes = f.read()
            
            # Try to get the PDF name and size
            file_name = os.path.basename(file.name)
            file_size = round(len(pdf_bytes) / 1024, 2)  # Size in KB
            
            # Save reference for later
            processed_files.append({
                "name": file_name,
                "size": file_size,
                "content": base64.b64encode(pdf_bytes).decode('utf-8')
            })
            
        except Exception as e:
            print(f"Error processing PDF {file.name}: {e}")
            return None, f"❌ Error processing {file.name}: {str(e)}"
    
    # Update global list of PDFs
    UPLOADED_PDFS = processed_files
    
    # Create status message
    if len(processed_files) == 1:
        status = f"📄 Uploaded: {processed_files[0]['name']} ({processed_files[0]['size']} KB)"
    else:
        status = f"📄 Uploaded {len(processed_files)} documents: " + ", ".join([pdf['name'] for pdf in processed_files])
    
    return files, status

def handle_question_and_documents(question: str, pdf_status: str) -> tuple:
    """Handle the user's query and process the uploaded PDFs."""
    global SESSION_ID, UPLOADED_PDFS
    
    if not question.strip():
        return [{"role": "user", "content": "Please enter a valid question."}], ""
    
    # Prepare PDF content for the API call
    pdf_contents = [pdf["content"] for pdf in UPLOADED_PDFS] if UPLOADED_PDFS else []

    # Prepare the request to the FastAPI server
    payload = {
        "question": question,
        "pdf_contents": pdf_contents,
        "session_id": SESSION_ID
    }
    
    try:
        # Send the request to the FastAPI server
        print(f"Sending request to FastAPI with {len(pdf_contents)} PDFs for session {SESSION_ID}")
        response = requests.post(f"{FASTAPI_URL}/ask", json=payload, timeout=300)
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        # IMPORTANT: Always update the session ID with what the server returned
        if "session_id" in result:
            server_session_id = result["session_id"]
            if server_session_id != SESSION_ID:
                print(f"⚠️ Updating session ID from {SESSION_ID} to server's {server_session_id}")
                SESSION_ID = server_session_id
        
        # Return the conversation history and empty the input
        return result.get("conversation", []), ""
    
    except requests.exceptions.RequestException as e:
        return [
            {
                "role": "user", 
                "content": f"Error communicating with backend server: {str(e)}"
            }
        ], question

def create_new_session():
    """Create a new session."""
    global SESSION_ID, UPLOADED_PDFS
    old_session_id = SESSION_ID
    SESSION_ID = str(uuid.uuid4())
    UPLOADED_PDFS = []

    try:
        payload = {
            "session_id": old_session_id
        }
        
        response = requests.delete(f"{FASTAPI_URL}/session", json=payload)
        if response.status_code == 200:
            print(f"Successfully deleted old session: {old_session_id}")
    except Exception as e:
        print(f"Error deleting old session: {e}")
    
    return (
        None, 
        "No documents uploaded",  
        [] 
    )

def clear_pdfs():
    """Clear uploaded PDFs and their embeddings."""
    global UPLOADED_PDFS
    
    if not UPLOADED_PDFS:
        print("No PDFs to clear")
        return None, "No documents uploaded"
    

    pdf_names = [pdf["name"] for pdf in UPLOADED_PDFS]
    UPLOADED_PDFS = []
    
    # Request server to delete all embeddings for this session
    try:
        print(f"Requesting deletion of document embeddings for session {SESSION_ID}")
        
        payload = {
            "session_id": SESSION_ID
        }
        
        response = requests.delete(f"{FASTAPI_URL}/documents", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Server response: {result.get('message', '')}")
        else:
            print(f"Error response: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error requesting deletion of document embeddings: {e}")
    
    return None, "No documents uploaded"

def cleanup_on_exit():
    """Clean up on exit - delete the session."""
    try:
        payload = {
            "session_id": SESSION_ID
        }
        
        response = requests.delete(f"{FASTAPI_URL}/session", json=payload)
        if response.status_code == 200:
            print(f"Successfully cleaned up session on exit: {SESSION_ID}")
    except Exception as e:
        print(f"Error cleaning up on exit: {e}")

# Register the cleanup function to run on exit
atexit.register(cleanup_on_exit)

# Create the Gradio interface - using components compatible with older Gradio versions
with gr.Blocks(css=custom_css) as iface:
    if logo_base64:
        logo_html = f'<div class="logo-container"><img src="data:image/svg+xml;base64,{logo_base64}" alt="Logo" style="height:60px;"><div class="tagline">Document Intelligence</div></div>'
    else:
        logo_html = '<div class="logo-container"><div class="logo-placeholder">SYNAPXE</div><div class="tagline">Document Intelligence</div></div>'

    gr.HTML(logo_html)
    
    # Main content
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation", 
                elem_id="chatbot",
                height=500,
                show_label=False,
                type="messages"  # Use message format to avoid deprecation warning
            )
            
            with gr.Row():
                question_input = gr.Textbox(
                    label="Ask a question:",
                    elem_id="question",
                    placeholder="Type your question here...",
                    lines=1,
                    show_label=False
                )
                submit_button = gr.Button("Send", elem_id="submit")
        
        with gr.Column(scale=1):
            # Use HTML for styling instead of Box component
            gr.HTML('<div class="upload-container">')
            gr.Markdown("### Document Upload")
            pdf_upload = gr.File(
                label="Upload PDF Documents",
                file_types=[".pdf"],
                file_count="multiple"
            )
            pdf_status = gr.Textbox(
                label="Document Status",
                placeholder="No documents uploaded",
                interactive=False,
                elem_id="status-indicator"
            )
            
            with gr.Row():
                clear_button = gr.Button("Clear Documents", elem_classes=["action-btn"])
                new_session_button = gr.Button("New Session", elem_classes=["action-btn"])
            gr.HTML('</div>')  # Close the upload container div


    # Set the action for the PDF upload
    pdf_upload.change(
        process_pdfs,
        inputs=[pdf_upload],
        outputs=[pdf_upload, pdf_status]
    )
    
    # Set the action for the clear button
    clear_button.click(
        clear_pdfs,
        inputs=[],
        outputs=[pdf_upload, pdf_status]
    )
    
    # Set the action for the new session button
    new_session_button.click(
        create_new_session,
        inputs=[],
        outputs=[pdf_upload, pdf_status, chatbot]
    )
    
    # Set the action for the submit button
    submit_button.click(
        handle_question_and_documents,
        inputs=[question_input, pdf_status],
        outputs=[chatbot, question_input]
    )
    
    # Also submit when pressing Enter in the question input
    question_input.submit(
        handle_question_and_documents,
        inputs=[question_input, pdf_status],
        outputs=[chatbot, question_input]
    )

# Check collections at startup
debug_collections()

# Launch the app
if __name__ == "__main__":
    iface.launch()