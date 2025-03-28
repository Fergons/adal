from pathlib import Path
import orjson
import shutil
from itertools import islice
from FlagEmbedding import BGEM3FlagModel





def create_embeddings(file_path, batch_size=64):
    file_path = Path(file_path)
    tmp_path = file_path.with_suffix(".tmp")
    model = BGEM3FlagModel(model_name_or_path="BAAI/bge-m3", devices="cuda:0")
    
    with open(file_path, "rb") as in_file, open(tmp_path, "wb") as out_file:
        while True:
            # Read a batch (list) of lines from the input file.
            batch_lines = list(islice(in_file, batch_size))
            if not batch_lines:
                break  # End of file reached.
            # Parse each JSON line into a record.
            records = [orjson.loads(line) for line in batch_lines]
            # Extract the text fields to create embeddings.
            texts = [record["text"] for record in records]
            # Generate embeddings in batch.
            embeddings = model.encode(texts, batch_size=batch_size, return_dense=True)["dense_vecs"]
            # Attach the embedding to its corresponding record and write updated record.
            for record, emb in zip(records, embeddings):
                # Convert to list if the embedding is e.g. a NumPy array.
                if hasattr(emb, "tolist"):
                    record["embedding"] = emb.tolist()
                else:
                    record["embedding"] = emb
                out_file.write(orjson.dumps(record) + b'\n')
    # Replace the old file with the new file after processing.
    shutil.move(tmp_path, file_path)

if __name__ == "__main__":
    create_embeddings("lit20.jsonl")












