from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import PyPDF2
import base64
import io
import uuid
import uvicorn
from typing import List, Dict, Any, Optional
from core.query import query_documents
from core.generate import (
    generate_QUESTIONFROMDOC,
    generate_SUMMARIZATION,
    detect_task,
)
from core.document_utils import split_text
from core.embedding import ollama_embeddings
from chromadb_setup import setup_chromadb, cleanup_chromadb, list_collections
import logging
import os
import atexit

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLaMA-based RAG System",
    description="A FastAPI based RAG system using LLaMA model and ChromaDB - Session-based PDF Only Mode with Multiple PDF Support",
    version="1.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session management - store conversations by session ID
CONVERSATIONS = {}

class Question(BaseModel):
    question: str
    pdf_contents: Optional[List[str]] = None
    session_id: Optional[str] = None

class Message(BaseModel):
    role: str
    content: str

class ConversationResponse(BaseModel):
    session_id: str
    conversation: List[Dict[str, str]]

class DeleteSessionRequest(BaseModel):
    session_id: str

class DeleteDocumentsRequest(BaseModel):
    session_id: str
    pdf_names: Optional[List[str]] = None
    force_clear: Optional[bool] = False

@app.on_event("startup")
async def startup_event():
    """Cleanup on server startup"""
    logger.info("Server starting up - cleaning up all existing collections")
    try:
        # Get existing collections for logging
        collections = list_collections()
        if collections:
            logger.info(f"Found existing collections: {collections}")
            
        # Use the cleanup function to delete all collections
        cleanup_chromadb()
        
        # Verify all collections were deleted
        collections_after = list_collections()
        if collections_after:
            logger.warning(f"Some collections still exist after cleanup: {collections_after}")
        else:
            logger.info("All collections successfully deleted on startup")
    except Exception as e:
        logger.error(f"Error during startup cleanup: {e}")


def get_session_id(request: Request, question: Question):
    """Get or create a session ID for the conversation."""
    if question.session_id:
        return question.session_id
    
    # Create a new session ID if none provided
    return str(uuid.uuid4())

def get_conversation(session_id: str) -> List[Dict[str, str]]:
    """Get the conversation for a given session ID or create a new one."""
    if session_id not in CONVERSATIONS:
        CONVERSATIONS[session_id] = []
    return CONVERSATIONS[session_id]

def process_pdf(file_content):
    """Extract text from the uploaded PDF file with improved formatting."""
    text = ""
    try:
        pdf_bytes = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_bytes)
        
        # Check if PDF is encrypted
        if pdf_reader.is_encrypted:
            try:
                # Try with empty password
                pdf_reader.decrypt('')
            except:
                raise ValueError("PDF is encrypted and could not be decrypted")
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            
            # Clean up the extracted text - remove excessive spaces
            # This regex replaces multiple spaces with a single space
            import re
            page_text = re.sub(r'\s+', ' ', page_text)
            
            text += page_text + "\n"
        
        logger.info(f"Processed PDF: {len(text)} characters extracted")
        return text
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise ValueError(f"Error processing PDF: {str(e)}")

def clean_chunk_text(text):
    import re
    
    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r'(\$?\s*(\d)\s*(\d)\s*(\.\s*(\d)\s*(\d)))', 
                  lambda m: f"{m.group(2)}{m.group(3)}.{m.group(5)}{m.group(6)}", 
                  text)
    
    text = re.sub(r'\b[a-zA-Z]\b', '', text)
    
    replacements = {
        'S G G S T': 'SG GST',
        'G S T': 'GST',
        'N B': 'NB',
        ' l t ': ' ',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    text = re.sub(r'([.,!?]){2,}', r'\1', text)
    
    text = re.sub(r'[^\w\s.,\-$()]', '', text)
    
    return ' '.join(text.split()).strip()

async def task(query: str, collection, pdf_files: bool, conversation: List) -> str:
    """Determine the task and generate an appropriate response - PDF Only Mode."""
    logger.info(f"Processing query: {query}")
    category = detect_task(str(conversation), query)
    logger.info(f"Detected task category: {category}")
    
    # Check if PDF is available
    if not pdf_files:
        return "Please upload PDF documents to answer questions. This system is configured to work with documents only."
    
    # Query relevant chunks regardless of task type
    relevant_chunks = query_documents(collection, query, n_results=4)
    
    # If no relevant chunks found
    if not relevant_chunks or len(relevant_chunks) == 0:
        return "I couldn't find relevant information in the uploaded documents to answer your question. Please try asking something related to the content of the documents."
    
    # Determine if summarization or question
    if "SUMMARIZATION" in category:
        respond = generate_SUMMARIZATION(query, relevant_chunks, conversation)
    else:
        # Default to document Q&A for all other categories
        respond = generate_QUESTIONFROMDOC(query, relevant_chunks, conversation)
    
    return respond

@app.get("/session-collections")
async def get_session_collections():
    """Get all session collections for initialization"""
    collections = list_collections()
    session_ids = collections
    session_collections = collections
    
    return {
        "collections": collections,
        "session_collections": session_collections,
        "session_ids": session_ids
    }

@app.get("/diagnostics")
async def diagnostics():
    """Check system diagnostics"""
    import os
    
    # Check ChromaDB directory
    chroma_path = "./chroma_db"
    
    try:
        # Check if directory exists
        dir_exists = os.path.exists(chroma_path)
        is_dir = os.path.isdir(chroma_path) if dir_exists else False
        
        # Check directory contents
        dir_contents = os.listdir(chroma_path) if is_dir else []
        
        # Check collections
        collections = list_collections()
        
        return {
            "directory_exists": dir_exists,
            "is_directory": is_dir,
            "directory_contents": dir_contents,
            "collections": collections
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask", response_model=ConversationResponse)
async def ask(
    question: Question,
    background_tasks: BackgroundTasks,
    session_id: str = Depends(get_session_id)
):
    """API endpoint to handle the user's query and multiple PDF uploads."""
    if not question.question or not question.question.strip():
        raise HTTPException(status_code=400, detail="Please enter a valid question")
    
    # Log the session ID for debugging
    logger.info(f"Processing request for session: {session_id}")
    
    # Get or create conversation for this session
    conversation = get_conversation(session_id)
    
    # Setup session-specific ChromaDB collection
    try:
        collection = setup_chromadb(session_id=session_id)
        logger.info(f"Collection for session {session_id} is ready")
    except Exception as e:
        logger.error(f"Error setting up ChromaDB: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    pdf_processed_before = True
    
    try:

        test_query = collection.query(
            query_texts=["test"],
            n_results=1
        )
        if not test_query or not test_query.get("documents") or len(test_query.get("documents")[0]) == 0:
            pdf_processed_before = False
            logger.info(f"No documents found for session {session_id}, will process PDFs if provided")
    except Exception as e:
        pdf_processed_before = False
        logger.info(f"Error checking for existing documents: {e}")
    
    # This is the updated snippet for the PDF processing section in run_fastapi.py
    # Find the section that processes PDFs in the /ask endpoint handler and replace it with this code:

    # Process PDFs if provided and not processed before
    pdf_processed = pdf_processed_before
    if question.pdf_contents and len(question.pdf_contents) > 0 and not pdf_processed_before:
        try:
            successful_pdfs = 0
            for i, pdf_content in enumerate(question.pdf_contents):
                try:
                    pdf_data = base64.b64decode(pdf_content)
                    doc = process_pdf(pdf_data)
                    chunks = split_text(doc)
                    
                    logger.info(f"Processing PDF #{i+1} with {len(chunks)} chunks for session {session_id}")
                    
                    # Process chunks immediately
                    added_chunks = 0
                    for j, chunk in enumerate(chunks):
                        # Clean up the chunk text before embedding
                        cleaned_chunk = clean_chunk_text(chunk)
                        
                        # Log original vs cleaned for debugging
                        if cleaned_chunk != chunk:
                            logger.debug(f"Cleaned chunk {j+1}:\nBefore: {chunk[:50]}...\nAfter: {cleaned_chunk[:50]}...")
                        
                        # Generate embedding
                        embedding = ollama_embeddings(cleaned_chunk)
                        
                        # Critical: Check if embedding is a valid list of floats
                        if isinstance(embedding, list) and len(embedding) > 0:
                            document_id = f"session_{session_id}_pdf{i+1}_chunk{j+1}"
                            try:
                                collection.upsert(
                                    ids=[document_id],
                                    documents=[cleaned_chunk],  # Use cleaned text
                                    embeddings=[embedding]
                                )
                                added_chunks += 1
                            except Exception as e:
                                logger.error(f"Error upserting chunk {j+1}: {str(e)}")
                        else:
                            logger.error(f"Invalid embedding received for chunk {j+1}: {embedding}")
                    
                    logger.info(f"Successfully added {added_chunks} chunks out of {len(chunks)} from PDF #{i+1}")
                    if added_chunks > 0:
                        successful_pdfs += 1
                        pdf_processed = True
                        
                except Exception as e:
                    logger.error(f"Error processing PDF #{i+1}: {str(e)}")
            
            if successful_pdfs > 0:
                logger.info(f"PDF processing complete for session {session_id} - {successful_pdfs} PDFs processed successfully")
            elif not pdf_processed_before:  # Only raise error if we haven't processed PDFs before
                raise HTTPException(status_code=400, detail="None of the PDFs could be processed successfully")
        except Exception as e:
            logger.error(f"Error processing PDFs: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing PDFs: {str(e)}")


    # Generate response
    response = await task(question.question, collection, pdf_processed or pdf_processed_before, conversation)
    
    # Update conversation
    conversation.append({"role": "user", "content": question.question})
    conversation.append({"role": "assistant", "content": response})
    
    # Return the session ID with the response
    return ConversationResponse(
        session_id=session_id,
        conversation=conversation
    )

@app.delete("/documents")
async def delete_documents(request: DeleteDocumentsRequest):
    """API endpoint to delete document embeddings for a session."""
    session_id = request.session_id
    
    logger.info(f"Document deletion requested for session {session_id}")
    
    try:
        # Check if collection exists first
        collections = list_collections()
        
        # Get collection names more reliably
        collection_names = []
        for coll in collections:
            try:
                if hasattr(coll, 'name'):
                    collection_names.append(coll.name)
                elif isinstance(coll, dict) and 'name' in coll:
                    collection_names.append(coll['name'])
                elif isinstance(coll, str):
                    collection_names.append(coll)
                else:
                    collection_names.append(str(coll))
            except Exception as e:
                logger.error(f"Error extracting collection name: {e}")
                continue
        
        logger.info(f"Current collections: {collection_names}")
        
        if session_id in collection_names:
            # Delete the collection
            success = cleanup_chromadb(session_id=session_id)
            
            # Verify deletion was successful - use improved collection name extraction
            collections_after = list_collections()
            collection_names_after = []
            for coll in collections_after:
                try:
                    if hasattr(coll, 'name'):
                        collection_names_after.append(coll.name)
                    elif isinstance(coll, dict) and 'name' in coll:
                        collection_names_after.append(coll['name'])
                    elif isinstance(coll, str):
                        collection_names_after.append(coll)
                    else:
                        collection_names_after.append(str(coll))
                except Exception as e:
                    continue
            
            if session_id not in collection_names_after:
                logger.info(f"Successfully deleted collection for session {session_id}")
                return {"success": True, "message": f"Cleared all documents for session {session_id}"}
            else:
                # Try a more forceful approach
                logger.warning(f"First attempt to delete collection {session_id} failed, trying alternative approach")
                try:
                    # Try to directly access the client and delete
                    chroma_client = chromadb.PersistentClient(path="./chroma_db")
                    chroma_client.delete_collection(name=session_id)
                    logger.info(f"Second attempt to delete collection {session_id} succeeded")
                    return {"success": True, "message": f"Cleared all documents for session {session_id} (second attempt)"}
                except Exception as e:
                    logger.error(f"Second attempt to delete collection {session_id} failed: {e}")
                    return {"success": False, "error": f"Collection still exists after two deletion attempts: {str(e)}"}
        else:
            logger.info(f"No collection found for session {session_id}")
            return {"success": True, "message": f"No documents found for session {session_id}"}
    except Exception as e:
        logger.error(f"Error deleting document embeddings: {e}")
        return {"success": False, "error": str(e)}

@app.get("/chroma-diagnostics")
async def chroma_diagnostics():
    """Check the current state of ChromaDB"""
    try:
        # Create a ChromaDB client
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        collections = list_collections()
        collection_info = []
        
        for collection in collections:
            try:
                name = None
                if hasattr(collection, 'name'):
                    name = collection.name
                elif isinstance(collection, dict) and 'name' in collection:
                    name = collection['name']
                elif isinstance(collection, str):
                    name = collection
                else:
                    name = str(collection)
                
                # Try to get collection count
                count = None
                try:
                    coll_obj = chroma_client.get_collection(name=name)
                    count_result = coll_obj.count()
                    count = count_result
                except Exception as e:
                    count = f"Error getting count: {str(e)}"
                
                collection_info.append({
                    "name": name,
                    "count": count
                })
            except Exception as e:
                collection_info.append({
                    "name": str(collection),
                    "error": str(e)
                })
        
        return {
            "collections": collection_info,
            "path": "./chroma_db"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/diagnostics")
async def diagnostics():
    """Check system diagnostics"""
    import os
    
    # Check ChromaDB directory
    chroma_path = "./chroma_db"
    
    try:
        # Check if directory exists
        dir_exists = os.path.exists(chroma_path)
        is_dir = os.path.isdir(chroma_path) if dir_exists else False
        
        # Check directory contents
        dir_contents = os.listdir(chroma_path) if is_dir else []
        
        # Check collections
        collections = list_collections()
        
        return {
            "directory_exists": dir_exists,
            "is_directory": is_dir,
            "directory_contents": dir_contents,
            "collections": collections
        }
    except Exception as e:
        return {"error": str(e)}

@app.delete("/session")
async def delete_session(request: DeleteSessionRequest):
    """API endpoint to delete a session and its associated embeddings."""
    session_id = request.session_id
    
    # Delete conversation
    if session_id in CONVERSATIONS:
        del CONVERSATIONS[session_id]
        logger.info(f"Deleted conversation for session: {session_id}")
    
    # Clean up session's ChromaDB collection
    cleanup_chromadb(session_id=session_id)
    logger.info(f"Cleaned up embeddings for session: {session_id}")
    
    return {"success": True, "message": f"Session {session_id} deleted successfully"}

@app.get("/list-collections")
async def get_collections():
    """List all collections in ChromaDB (for debugging purposes)"""
    collections = list_collections()
    return {"collections": collections}

@app.on_event("shutdown")
def shutdown_event():
    """Cleanup on server shutdown"""
    logger.info("Server shutting down - cleaning up all collections")
    try:
        # Get existing collections for logging
        collections = list_collections()
        if collections:
            logger.info(f"Found collections to clean up: {collections}")
            
        # Use the cleanup function to delete all collections
        cleanup_chromadb()
        
        # No verification needed on shutdown
        logger.info("Cleanup complete on shutdown")
    except Exception as e:
        logger.error(f"Error during shutdown cleanup: {e}")

if __name__ == "__main__":
    # Register cleanup function to run on exit
    atexit.register(cleanup_chromadb)
    uvicorn.run(app, host="0.0.0.0", port=8000)