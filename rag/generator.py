"""Answer generation via Gemini, grounded in retrieved chunks."""

import google.generativeai as genai

MODEL_NAME = "gemini-2.0-flash"

PROMPT_TEMPLATE = """You are answering questions using only the context below, \
which was retrieved from the user's own notes/PDFs. If the context does not \
contain the answer, say so plainly instead of guessing.

Context:
{context}

Question: {question}

Answer, and cite the source file(s) you used in brackets, e.g. [source.pdf]."""


def configure(api_key: str):
    genai.configure(api_key=api_key)


def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[{c['source']} chunk {c['chunk_id']}]\n{c['text']}")
    return "\n\n".join(parts)


def generate_answer(question: str, chunks: list[dict]) -> str:
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = PROMPT_TEMPLATE.format(context=build_context(chunks), question=question)
    response = model.generate_content(prompt)
    return response.text
