"""
Mistral + Sentence-Transformers client utilities.

- Mistral Chat API: chat(prompt) -> str
- Embeddings: generate_embeddings(texts) -> List[List[float]]
- Query embedding: embed_query(text) -> List[float]

Environment Variables:
    MISTRAL_API_KEY
    LLM_MODEL (default: mistral-small-latest)
"""

from __future__ import annotations
import os
import json
import time
import math
from typing import List, Iterable, Optional
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -----------------------
# CONFIG
# -----------------------
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("❌ MISTRAL_API_KEY not found in .env file!")

MISTRAL_BASE_URL = "https://api.mistral.ai/v1/chat/completions"
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-small-latest")

# Requests session
_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """Return global requests.Session (lazy)."""
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


# -----------------------
# CHAT COMPLETION (MISTRAL)
# -----------------------
def chat(prompt: str, max_tokens: int = 512, temperature: float = 0.2, timeout: int = 120) -> str:
    """
    Send prompt to Mistral Chat Completion API and return the response text.
    """

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    sess = _get_session()
    last_exc = None

    for attempt in range(3):
        try:
            r = sess.post(MISTRAL_BASE_URL, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()

            data = r.json()
            return data["choices"][0]["message"]["content"]

        except requests.HTTPError as he:
            last_exc = he
            status = he.response.status_code
            if status in (429,) or (500 <= status < 600):
                time.sleep(1.5 + attempt)
                continue
            raise

        except Exception as e:
            last_exc = e
            time.sleep(1 + attempt)
            continue

    raise RuntimeError(f"Mistral request failed after retries: {last_exc}")


# -----------------------
# EMBEDDINGS (SentenceTransformers)
# -----------------------
_SBERT = None


def _ensure_sbert_loaded(model_name: Optional[str] = None):
    """Lazy load SentenceTransformer."""
    global _SBERT
    if _SBERT is not None:
        return _SBERT

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers is not installed. Install with: pip install sentence-transformers"
        ) from e

    model_name = model_name or os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2")

    try:
        _SBERT = SentenceTransformer(model_name, device="cpu")
        return _SBERT
    except Exception as e:
        raise RuntimeError(f"Failed to load SBERT model '{model_name}': {e}") from e


def _chunk_iterable(iterable: Iterable, size: int):
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(size):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            break
        yield chunk


def generate_embeddings(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """Return embeddings for list of texts."""
    if not texts:
        return []

    model = _ensure_sbert_loaded()
    all_embs = []

    for chunk in _chunk_iterable(texts, batch_size):
        embs = model.encode(chunk, show_progress_bar=False, convert_to_numpy=True)
        for row in embs:
            all_embs.append(list(row))

    return all_embs


def embed_query(text: str) -> List[float]:
    """Return embedding for a single query."""
    if not text:
        return []
    embs = generate_embeddings([text], batch_size=1)
    return embs[0] if embs else []


# -----------------------
# TEST
# -----------------------
if __name__ == "__main__":
    print("Testing Mistral chat...")
    print(chat("Hello! Explain RAG in simple words."))
    print("Testing embeddings...")
    vec = embed_query("machine learning")
    print(len(vec), "dimensions")
