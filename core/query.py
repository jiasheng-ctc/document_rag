def query_documents(collection, question: str, n_results: int = 2) -> list:
    """
    Query the ChromaDB collection for documents relevant to the given question.

    Args:
        question (str): The input question or query text.
        n_results (int, optional): The number of relevant documents to retrieve. Default is 2.

    Returns:
        list: A list of relevant document chunks.
    """
    print(
        f"🔍 Querying for relevant documents to the question: '{question}' (Top {n_results} results)"
    )

    # Query the collection
    results = collection.query(query_texts=question, n_results=n_results)

    # Extract relevant document chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]

    print(f"✅ Returning {len(relevant_chunks)} relevant document chunks.")
    return relevant_chunks
