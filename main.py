"""Bare-metal RAG over your own notes/PDFs.

Usage:
    python main.py ingest          # build the index from notes/
    python main.py ask "question"  # ask one question
    python main.py chat            # interactive chat loop
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from rag.chunker import chunk_documents
from rag.embedder import Embedder
from rag.generator import configure, generate_answer
from rag.loader import load_documents
from rag.store import VectorStore

NOTES_DIR = Path(__file__).parent / "notes"
INDEX_DIR = Path(__file__).parent / "index"
TOP_K = 4


def ingest():
    if not NOTES_DIR.exists() or not any(NOTES_DIR.iterdir()):
        print(f"No files found in {NOTES_DIR}. Add some .pdf/.md/.txt files first.")
        return

    print(f"Loading documents from {NOTES_DIR} ...")
    documents = load_documents(NOTES_DIR)
    if not documents:
        print("No supported non-empty documents found in notes/.")
        return
    print(f"Loaded {len(documents)} document(s).")

    chunks = chunk_documents(documents)
    if not chunks:
        print("No chunks produced (documents may be empty).")
        return
    print(f"Split into {len(chunks)} chunk(s).")

    print("Embedding chunks (first run downloads the model, ~90MB) ...")
    embedder = Embedder()
    vectors = embedder.embed([c["text"] for c in chunks])

    store = VectorStore()
    store.add(vectors, chunks)
    store.save(INDEX_DIR)
    print(f"Index saved to {INDEX_DIR}")


def _load_store_and_key():
    if not (INDEX_DIR / "vectors.npy").exists():
        print("No index found. Run `python main.py ingest` first.")
        sys.exit(1)

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)

    configure(api_key)
    store = VectorStore.load(INDEX_DIR)
    embedder = Embedder()
    return store, embedder


def ask(question: str):
    store, embedder = _load_store_and_key()
    query_vector = embedder.embed_one(question)
    chunks = store.search(query_vector, top_k=TOP_K)

    print("\nRetrieved chunks:")
    for c in chunks:
        print(f"  [{c['score']:.3f}] {c['source']} chunk {c['chunk_id']}")

    answer = generate_answer(question, chunks)
    print(f"\nAnswer:\n{answer}")


def chat():
    store, embedder = _load_store_and_key()
    print("Chat with your notes. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        query_vector = embedder.embed_one(question)
        chunks = store.search(query_vector, top_k=TOP_K)
        answer = generate_answer(question, chunks)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    if command == "ingest":
        ingest()
    elif command == "ask":
        if len(sys.argv) < 3:
            print("Usage: python main.py ask \"your question\"")
            sys.exit(1)
        ask(sys.argv[2])
    elif command == "chat":
        chat()
    else:
        print(__doc__)
        sys.exit(1)
