"""
src/ingest.py
=============
Loads all .txt files from knowledge_base/, chunks them,
embeds with sentence-transformers, and stores in ChromaDB.

Run ONCE after extract_pdfs.py, then run again if you add new docs.

Usage:
    uv run python src/ingest.py
"""

import json
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

KB_DIR          = Path("knowledge_base")
CHROMA_DIR      = Path("chroma_db")
COLLECTION_NAME = "taxmate_lk"
CHUNK_SIZE      = 400   # words per chunk — smaller = more precise retrieval
CHUNK_OVERLAP   = 60    # overlap keeps context across chunk boundaries


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        chunk = " ".join(words[start : start + size])
        if len(chunk.strip()) > 50:   # skip tiny chunks
            chunks.append(chunk)
        start += size - overlap
    return chunks


def load_documents() -> list[dict]:
    docs = []
    txt_files = sorted(KB_DIR.glob("*.txt"))
    # skip metadata file
    txt_files = [f for f in txt_files if f.name != "source_metadata.json"]

    print(f"Found {len(txt_files)} documents in {KB_DIR}/\n")

    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text) < 100:
            print(f"Skipping {txt_file.name} — too short ({len(text)} chars)")
            continue

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            docs.append({
                "id":       f"{txt_file.stem}_c{i}",
                "text":     chunk,
                "source":   txt_file.name,
                "chunk":    i,
                "total":    len(chunks),
            })
        print(f" {txt_file.name}: {len(chunks)} chunks")

    return docs


def build_store(docs: list[dict]) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"   # fast, local, no API key
    )

    # fresh rebuild each time
    try:
        client.delete_collection(COLLECTION_NAME)
        print("\nDeleted existing collection (rebuilding fresh)")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # batch insert
    BATCH = 64
    for i in tqdm(range(0, len(docs), BATCH), desc="Indexing"):
        batch = docs[i : i + BATCH]
        collection.add(
            ids=       [d["id"]     for d in batch],
            documents= [d["text"]   for d in batch],
            metadatas= [{
                "source":      d["source"],
                "chunk_index": d["chunk"],
                "total_chunks":d["total"],
            } for d in batch],
        )

    return collection


def sanity_check(collection: chromadb.Collection):

    test_queries = [
        "withholding tax photographers 2026",
        "15 percent cap foreign currency ITES",
        "quarterly installment payment deadline",
    ]
    print("\n Sanity check")
    for q in test_queries:
        results = collection.query(query_texts=[q], n_results=1)
        doc  = results["documents"][0][0][:120]
        src  = results["metadatas"][0][0]["source"]
        print(f"  Q: {q}")
        print(f"  A: [{src}] {doc}...\n")


def main():
    print("=" * 55)
    print("TaxMate LK — Building Vector Store")
    print("=" * 55 + "\n")

    docs = load_documents()
    if not docs:
        print("No documents found. Run extract_pdfs.py first.")
        return

    print(f"\nTotal chunks to index: {len(docs)}")
    collection = build_store(docs)
    print(f"\n {collection.count()} chunks stored in {CHROMA_DIR}/")
    sanity_check(collection)
    print("Next step: uv run python src/tools.py")


if __name__ == "__main__":
    main()
