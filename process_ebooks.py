from typing import Literal, get_args
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
import sys
import orjson
import re
import semchunk
from argparse import ArgumentParser
from pathlib import Path
from logging import getLogger

logger = getLogger(__name__)


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
        soup = BeautifulSoup(item.get_content(), features="xml")
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


def convert_epub_to_txt(epub_path: str) -> str:
    """
    Converts an EPUB file to a text file.
    """
    docs = extract_text_by_docs_from_epub(epub_path)
    return "\n\n".join(docs.values())


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

def mode_chunk(*, book_id: str, epub_path: str, max_chunk_length: int, overlap: float, **kwargs):
    book_chunks = chunk_epub(epub_path, book_id, max_chunk_length=max_chunk_length, overlap=overlap)
    with open(f"{book_id}.jsonl", "wb") as f:
        for chunk in book_chunks:
            f.write(orjson.dumps(chunk))
            f.write(b"\n")

def mode_convert(*, book_id: str, epub_path: str, **kwargs):
    text = convert_epub_to_txt(epub_path)
    with open(f"{book_id}.txt", "w", encoding="utf-8") as f:
        f.write(text)


def find_book_by_id(books_dir: str, book_id: str) -> str:
    for book_path in Path(books_dir).glob("*.epub"):
        if book_path.stem.split("_")[1] == book_id:
            return book_path
    raise ValueError(f"Book with id {book_id} not found in {books_dir}")


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
    parser.add_argument("--epub_path", type=str, default=None, help="Path to the EPUB file")
    parser.add_argument("--book_id", type=str, default=None, help="Book ID")
    parser.add_argument("--books_dir", type=str, default=None, help="Path to the directory containing the EPUB files")
    parser.add_argument("--mode", type=mode_type, default="convert", help="Mode to run the script in")
    parser.add_argument("--max_chunk_length", type=int, default=256, help="Maximum chunk length")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap between chunks")
    args = parser.parse_args()

    assert args.book_id is not None
    assert args.epub_path is not None or args.books_dir is not None

    if args.epub_path is None:
        args.epub_path = find_book_by_id(args.books_dir, args.book_id)
    assert Path(args.epub_path).is_file()
    
    if args.mode == "chunk":
        logger.info(f"Chunking {args.epub_path} into {args.book_id}.jsonl")
        mode_chunk(**vars(args))
    elif args.mode == "convert":
        logger.info(f"Converting {args.epub_path} to {args.book_id}.txt")
        mode_convert(**vars(args))
    logger.info("Done")

   
