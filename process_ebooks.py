from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
import sys
import orjson

def extract_text_from_epub(epub_path):
    """
    Extracts and concatenates text from all document items in the EPUB.
    """
    book = epub.read_epub(epub_path)
    full_text = ""
    
    # Iterate over all items in the EPUB and extract text from HTML documents.
    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            # Parse HTML content and extract text.
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text(separator=" ", strip=True)
            full_text += text + "\n"
    
    return full_text

def chunk_text(text, max_tokens=500):
    """
    Splits the given text into chunks where each chunk contains as many sentences 
    as possible such that the token count (computed using tiktoken) does not exceed max_tokens.
    
    The text is first split into sentences using nltk's Punkt sentence tokenizer.
    If a single sentence has more tokens than max_tokens, it is added as its own chunk.
    """
    # Import and download the required nltk tokenizer data.
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import sent_tokenize

    # Import tiktoken and get an encoder for a specific model.
    import tiktoken
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    # Split the text into sentences.
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(encoding.encode(sentence))
        
        # If a single sentence alone exceeds the token limit,
        # finish the current chunk (if any) and append the long sentence as its own chunk.
        if sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_tokens = 0
            chunks.append(sentence.strip())
            continue

        # Check if adding the sentence to the current chunk would exceed the token limit.
        if current_tokens + sentence_tokens <= max_tokens:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_tokens += sentence_tokens
        else:
            # If it would exceed the limit, append the current chunk to the list, and start a new one.
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = sentence_tokens

    # Append any remaining chunk.
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def process_epub(epub_path, book_id, max_chunk_length=500):
    """
    Processes the EPUB file, chunks its text, and returns a list of dictionaries.
    
    Each dictionary has:
        - "text": the text chunk.
        - "citation": a string combining book_id and the chunk number.
    
    Note: The max_chunk_length parameter now represents the maximum number of tokens per chunk.
    """
    # Extract the full text from the EPUB.
    text = extract_text_from_epub(epub_path)
    
    # Split the full text into chunks using the token-based chunk_text.
    chunks = chunk_text(text, max_tokens=max_chunk_length)
    
    # Build the list of dictionaries with citations.
    book_chunks = []
    for i, chunk in enumerate(chunks, start=1):
        citation = f"{book_id}_{i}"  # e.g., "book123_1", "book123_2", etc.
        book_chunks.append({
            "text": chunk,
            "citation": citation,
        })
    
    return book_chunks

# Example usage:
if __name__ == "__main__":
    epub_path = "lit20.epub"  # Path to your EPUB file.
    book_id = "lit20"         # Your desired book identifier.
    
    # Process the EPUB and get the chunks.
    book_chunks = process_epub(epub_path, book_id, max_chunk_length=100)
    
    # Print the results.
    with open(f"{book_id}.jsonl", "wb") as f:
        for chunk in book_chunks:
            f.write(orjson.dumps(chunk))
            f.write(b"\n")
