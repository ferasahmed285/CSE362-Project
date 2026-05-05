# rag/retriever.py

import os
import threading

KNOWLEDGE_BASE_PATH = "data/knowledge_base.txt"

_cache = {}
_cache_lock = threading.Lock()  # FIX 6: thread-safe cache access
_knowledge_chunks = None
_kb_lock = threading.Lock()  # FIX 6: thread-safe knowledge base loading


def load_knowledge_base():
    global _knowledge_chunks

    # Fast path: already loaded (read is safe without lock)
    if _knowledge_chunks is not None:
        return _knowledge_chunks

    with _kb_lock:
        # Double-check after acquiring lock
        if _knowledge_chunks is not None:
            return _knowledge_chunks

        if not os.path.exists(KNOWLEDGE_BASE_PATH):
            _knowledge_chunks = []
            return _knowledge_chunks

        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as file:
            text = file.read()

        # Split into sentences
        chunks = [chunk.strip() for chunk in text.split(".") if chunk.strip()]
        _knowledge_chunks = chunks
        return chunks


def retrieve_context(query):
    # Cache hit (thread-safe read)
    with _cache_lock:
        if query in _cache:
            return _cache[query]

    chunks = load_knowledge_base()

    if not chunks:
        return "No context available"

    query_words = set(query.lower().split())

    scored_chunks = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(query_words.intersection(chunk_words))
        scored_chunks.append((score, chunk))

    # Sort by relevance
    scored_chunks.sort(reverse=True, key=lambda x: x[0])

    top_chunks = [chunk for score, chunk in scored_chunks[:3]]

    context = ". ".join(top_chunks)

    if not context:
        context = chunks[0]

    # Thread-safe write
    with _cache_lock:
        _cache[query] = context
    return context