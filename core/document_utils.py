import os


def load_documents_from_directory(directory_path: str):
    """
    Load text documents from a specified directory.

    Args:
        directory_path (str): The path to the directory containing text files.

    Returns:
        list: A list of dictionaries where each dictionary contains 'id' (filename)
              and 'text' (file contents).
    """
    print(f"📂 Loading documents from directory: {directory_path}")
    documents = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            print(f"📄 Reading file: {filename}")

            with open(file_path, "r", encoding="utf-8") as file:
                documents.append({"id": filename, "text": file.read()})

    print(f"✅ Successfully loaded {len(documents)} documents.")
    return documents


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 20) -> list:
    """
    Split a large text into smaller chunks of a specified size with overlap.

    Args:
        text (str): The text to split into chunks.
        chunk_size (int, optional): The maximum size of each chunk. Default is 1000 characters.
        chunk_overlap (int, optional): The overlap between consecutive chunks. Default is 20 characters.

    Returns:
        list: A list of text chunks.
    """
    chunks = []
    start = 0

    print(
        f"🔀 Splitting text into chunks of size {chunk_size} with {chunk_overlap}-character overlap."
    )

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap

    print(f"✅ Text split into {len(chunks)} chunks.")
    return chunks
