import logging

logger = logging.getLogger(__name__)

def query_documents(collection, question: str, n_results: int = 4) -> list:
    """
    Query the ChromaDB collection for documents relevant to the given question.

    Args:
        collection: The ChromaDB collection to query
        question (str): The input question or query text.
        n_results (int, optional): The number of relevant documents to retrieve. Default is 4.

    Returns:
        list: A list of relevant document chunks.
    """
    logger.info(f"Querying for relevant documents to the question: '{question}' (Top {n_results} results)")

    try:
        # Query the collection
        results = collection.query(
            query_texts=question, 
            n_results=n_results,
            include=["documents", "distances", "metadatas"]
        )
        
        # Extract relevant document chunks with their IDs and distances
        relevant_chunks = []
        
        if results and "documents" in results and len(results["documents"]) > 0:
            # Flatten the list of documents
            for i, docs in enumerate(results["documents"]):
                for j, doc in enumerate(docs):
                    # Get the document ID and distance if available
                    doc_id = results.get("ids", [[]])[i][j] if "ids" in results else f"doc_{i}_{j}"
                    distance = results.get("distances", [[]])[i][j] if "distances" in results else None
                    
                    # Add the document to the list of relevant chunks
                    relevant_chunks.append(doc)
                    
                    # Log document details
                    doc_preview = doc[:50] + "..." if len(doc) > 50 else doc
                    if distance:
                        logger.info(f"Document {doc_id} (relevance: {1-distance:.4f}): {doc_preview}")
                    else:
                        logger.info(f"Document {doc_id}: {doc_preview}")
        
        logger.info(f"Found {len(relevant_chunks)} relevant document chunks")
        return relevant_chunks
        
    except Exception as e:
        logger.error(f"Error querying documents: {e}")
        return []