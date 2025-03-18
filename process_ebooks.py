from typing import Literal, get_args
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
import sys
import orjson
import re
import semchunk
from argparse import ArgumentParser
from pathlib import Path


def extract_text_by_docs_from_epub(epub_path: str|Path) -> dict[str, str]:
    """
    Extracts and concatenates text from all document items in the EPUB.
    """
    epub_path = Path(epub_path)
    book = epub.read_epub(epub_path.as_posix())
    docs = {}
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        if not item.is_chapter():
            continue
        soup = BeautifulSoup(item.get_body_content(), features="xml")
        text = soup.get_text(separator=" ", strip=True)
        if len(text) < 20:
            continue
        docs[item.get_name()] = text
    return dict(sorted(docs.items(), key=lambda x: int(re.search(r'\d+', x[0]).group()) if re.search(r'\d+', x[0]) else 0))


def chunk_epub(epub_path: str, book_id: str, max_chunk_length: int = 200, overlap: float = 0.2) -> list[dict[str, str]]:
    """
    Processes the EPUB file, chunks its text, and returns a list of dictionaries.
    
    Each dictionary has:
        - "text": the text chunk.
        - "doc_id": the document id usually an epub file consists of multiple documents.
        - "chunk_id": the chunk id.
        - "book_id": the book id.
        - "citation_id": the citation id.
    
    Note: The max_chunk_length parameter now represents the maximum number of tokens per chunk.
    """
    chunker = semchunk.chunkerify("BAAI/bge-m3", chunk_size=max_chunk_length)
    docs = extract_text_by_docs_from_epub(epub_path)
    chunks_by_docs = chunker(docs.values(), overlap=overlap)
    book_chunks = []
    i = 0
    for doc_id, doc_chunks in zip(docs.keys(), chunks_by_docs):
        for chunk in doc_chunks:
            citation = f"{book_id}_{i}"  # e.g., "book123_1_1", "book123_2_2", etc.
            book_chunks.append({
                "text": chunk,
                "doc_id": doc_id,
                "max_chunk_size": max_chunk_length,
                "chunk_id": i,
                "book_id": book_id,
                "citation_id": citation,
            })
            i += 1
           
    return book_chunks


def convert_epub_to_txt(epub_path: str, book_id: str) -> None:
    """
    Converts an EPUB file to a text file.
    """
    docs = extract_text_by_docs_from_epub(epub_path)
    with open(f"{book_id}.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(docs.values()))


def load_jsonl(file_path):
    """Loads a JSONL file and returns a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(orjson.loads(line))
            except orjson.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
    return data


MODE = Literal["chunk", "convert"]

def mode_type(value):
    if value not in get_args(MODE):
        raise ValueError(f"Invalid mode: {value}. Choose from {get_args(MODE)}")
    return value


if __name__ == "__main__":
    # test purposes
    epub_path = "lit20.epub"
    book_id = "lit20"        
    
    parser = ArgumentParser()
    parser.add_argument("--epub_path", type=str, default="lit20.epub")
    parser.add_argument("--book_id", type=str, default="lit20")
    parser.add_argument("--mode", type=mode_type, default="convert")
    parser.add_argument("--max_chunk_length", type=int, default=256)
    parser.add_argument("--overlap", type=float, default=0.2)
    args = parser.parse_args()
    
    if args.mode == "chunk":
        book_chunks = chunk_epub(args.epub_path, args.book_id, max_chunk_length=args.max_chunk_length, overlap=args.overlap)
        with open(f"{args.book_id}.jsonl", "wb") as f:
            for chunk in book_chunks:
                f.write(orjson.dumps(chunk))
                f.write(b"\n")

    elif args.mode == "convert":
        convert_epub_to_txt(args.epub_path, args.book_id)
       

   
