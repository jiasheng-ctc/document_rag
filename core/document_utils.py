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

def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Split a large text into smaller chunks of a specified size with overlap.
    Uses smaller chunks and more overlap for better context preservation.

    Args:
        text (str): The text to split into chunks.
        chunk_size (int, optional): The maximum size of each chunk. Default is 500 characters (reduced from 1000).
        chunk_overlap (int, optional): The overlap between consecutive chunks. Default is 50 characters (increased from 20).

    Returns:
        list: A list of text chunks.
    """
    chunks = []
    start = 0

    print(
        f"🔀 Splitting text into chunks of size {chunk_size} with {chunk_overlap}-character overlap."
    )

    # Break on sentence boundaries when possible
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed the chunk size
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            # Add the current chunk to our list
            chunks.append(current_chunk)
            # Start a new chunk with overlap
            last_period = current_chunk.rfind(".")
            if last_period != -1 and last_period > len(current_chunk) - chunk_overlap:
                # If we can find a period in the overlap region, start from there
                overlap_text = current_chunk[last_period+1:]
                current_chunk = overlap_text + sentence
            else:
                # Otherwise take the last chunk_overlap characters
                current_chunk = current_chunk[-chunk_overlap:] + sentence
        else:
            # Add the sentence to the current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    print(f"✅ Text split into {len(chunks)} chunks with improved boundary handling.")
    return chunks