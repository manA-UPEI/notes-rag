"""Word-based fixed-size chunking with overlap.

This is the simplest possible chunking strategy. It ignores sentence/paragraph
boundaries on purpose so you can see the baseline before improving on it.
"""


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_documents(documents: list[dict], chunk_size: int = 200, overlap: int = 40) -> list[dict]:
    """Turns [{"source", "text"}] into [{"source", "chunk_id", "text"}]."""
    chunks = []
    for doc in documents:
        for i, chunk in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            chunks.append({"source": doc["source"], "chunk_id": i, "text": chunk})
    return chunks
