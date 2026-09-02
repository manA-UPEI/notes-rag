"""Load raw text out of the notes/ folder. Supports .pdf, .md, .txt."""

from pathlib import Path

from pypdf import PdfReader


def load_documents(notes_dir: Path) -> list[dict]:
    """Returns a list of {"source": filename, "text": full_text} for every
    supported file in notes_dir."""
    documents = []
    for path in sorted(notes_dir.iterdir()):
        if path.suffix.lower() == ".pdf":
            text = _read_pdf(path)
        elif path.suffix.lower() in (".md", ".txt"):
            text = path.read_text(encoding="utf-8")
        else:
            continue
        if text.strip():
            documents.append({"source": path.name, "text": text})
    return documents


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)
