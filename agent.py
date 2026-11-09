import glob
import hashlib
import json
import math
import os

import ollama
from pypdf import PdfReader

DATA = "data"
MEMORY_FILE = "memory.json"
CHAT_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"


def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_memory(mem):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(mem, f)


def read_file(path):
    if path.lower().endswith(".pdf"):
        return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunks(text, size=900, overlap=150):
    i = 0
    while i < len(text):
        yield text[i:i + size]
        i += size - overlap


def embed(text):
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def ingest(mem):
    added = 0
    for f in glob.glob(f"{DATA}/**/*.*", recursive=True):
        if not f.lower().endswith((".pdf", ".txt", ".md")):
            continue
        for c in chunks(read_file(f)):
            if not c.strip():
                continue
            key = hashlib.md5(c.encode("utf-8")).hexdigest()
            if key in mem:
                continue
            mem[key] = {"text": c, "source": f, "embedding": embed(c)}
            added += 1
    save_memory(mem)
    print(f"Memory ready: {len(mem)} chunks ({added} new)")


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def ask(mem, question, top_k=4):
    q = embed(question)
    ranked = sorted(mem.values(), key=lambda m: cosine(q, m["embedding"]), reverse=True)
    context = "\n---\n".join(m["text"] for m in ranked[:top_k])
    messages = [
        {"role": "system", "content":
         "You are Usman's private assistant. Answer using ONLY the context. "
         "If the answer is not in the context, say you don't know. Never invent facts."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return ollama.chat(model=CHAT_MODEL, messages=messages)["message"]["content"]


if __name__ == "__main__":
    memory = load_memory()
    ingest(memory)
    while True:
        q = input("\nYou: ")
        if q.lower() in ("exit", "quit"):
            break
        print("\nAgent:", ask(memory, q))