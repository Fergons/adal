from argparse import ArgumentParser
from process_ebooks import convert_epub_to_txt, chunk_epub
from create_embeddings import create_embeddings
import tqdm
from pathlib import Path
import logging
import pandas as pd
import orjson

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def check_cache_hit(cache_dir: str, book_id: str):
    path_to_check = Path(cache_dir) / f"{book_id}.jsonl"
    if not path_to_check.exists():
        return False
    with open(path_to_check, "rb") as f:
        if f.read() == b"":
            return False
    return True


def prepare_tropes_dataset(book_ids: list[str], cache_dir: str, books_dir: str, mode: str):
    books_dir = Path(books_dir)
    cache_dir = Path(cache_dir)
    for book_id in tqdm.tqdm(book_ids, desc="Processing books"):
        book_path = list(books_dir.glob(f"*{book_id}.epub"))
        if not book_path:
            logger.warning(f"Book {book_id} not found in {books_dir}")
            continue
        book_path = book_path[0]
        if mode == "check":
            if check_cache_hit(cache_dir, book_id):
                logger.info(f"Cache hit for {book_id}")
                continue
            else:
                logger.info(f"Cache miss for {book_id}")
                continue
        if mode == "convert":
            text = convert_epub_to_txt(book_path)
            with open(cache_dir/f"{book_id}.txt", "w") as f:
                f.write(text)
        elif mode == "chunk":
            chunks = chunk_epub(book_path, book_id)
            with open(cache_dir/f"{book_id}.jsonl", "wb") as f:
                for chunk in chunks:
                    f.write(orjson.dumps(chunk))
                    f.write(b"\n")  
        elif mode == "embed":
            if check_cache_hit(cache_dir, book_id):
                logger.info(f"Cache hit for {book_id}")
                continue
            logger.debug(f"Chunking {book_id}")
            chunks = chunk_epub(book_path, book_id)
            logger.debug(f"Number of chunks: {len(chunks)}")
            with open(cache_dir/f"{book_id}.jsonl", "wb") as f:
                for chunk in chunks:
                    f.write(orjson.dumps(chunk))
                    f.write(b"\n")
            create_embeddings(cache_dir/f"{book_id}.jsonl")


if __name__ == "__main__":
    DEFAULT_CACHE_DIR = "preprocessed_books"
    DEFAULT_BOOKS_DIR = "D:/narana/data/tvtropes/plot_books"
    DEFAULT_MODE = "embed"
    DEFAULT_TVTROPES_CSV = "tropes_dataset.csv"

    parser = ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR, help="Cache directory")
    parser.add_argument("--books_dir", type=str, default=DEFAULT_BOOKS_DIR, help="Books directory")
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE, help="Mode - convert, chunk, embed")
    book_ids = pd.read_csv(DEFAULT_TVTROPES_CSV)["book_id"].unique().tolist()
    args = parser.parse_args()
    logger.info(f"Preparing tropes dataset for {len(book_ids)} books.")
    logger.info(f"Mode: {args.mode}")
    prepare_tropes_dataset(book_ids, args.cache_dir, args.books_dir, args.mode)
    logger.info(f"Done. Check the cache directory {args.cache_dir} for the preprocessed books.")