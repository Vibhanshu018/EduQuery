"""
Mistral client utilities.

- Mistral Chat API:       chat(prompt) -> str
- Mistral Embeddings API: generate_embeddings(texts) -> List[List[float]]
- Query embedding:        embed_query(text) -> List[float]

Replaces sentence-transformers/PyTorch (3-4 GB) with Mistral's embedding
API endpoint — keeping the Docker image under 300 MB on Railway free tier.

Environment Variables:
    MISTRAL_API_KEY  (required)
    LLM_MODEL        (default: mistral-small-latest)
    EMBED_MODEL      (default: mistral-embed)
"""

from __future__ import annotations
import os
import time
from typing import List, Iterable, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("❌ MISTRAL_API_KEY not found in environment / .env file!")

MISTRAL_CHAT_URL  = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_EMBED_URL = "https://api.mistral.ai/v1/embeddings"
LLM_MODEL         = os.getenv("LLM_MODEL",   "mistral-small-latest")
EMBED_MODEL       = os.getenv("EMBED_MODEL",  "mistral-embed")

# ── Shared session ────────────────────────────────────────────────────────────

_session: Optional[requests.Session] = None

def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
    return _session

def _auth_headers() -> dict:
    return {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }

# ── Chat completion ───────────────────────────────────────────────────────────

def chat(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    timeout: int = 120,
) -> str:
    """Send a prompt to Mistral Chat and return the response text."""
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    sess     = _get_session()
    last_exc = None

    for attempt in range(3):
        try:
            r = sess.post(MISTRAL_CHAT_URL, json=payload, headers=_auth_headers(), timeout=timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

        except requests.HTTPError as he:
            last_exc = he
            code = he.response.status_code
            if code == 429 or 500 <= code < 600:
                time.sleep(1.5 + attempt)
                continue
            raise

        except Exception as e:
            last_exc = e
            time.sleep(1 + attempt)
            continue

    raise RuntimeError(f"Mistral chat failed after retries: {last_exc}")

# ── Embeddings (Mistral API — no PyTorch, no sentence-transformers) ───────────

def _chunk_iterable(iterable: Iterable, size: int):
    """Yield successive chunks of `size` from an iterable."""
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


def generate_embeddings(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Return embeddings for a list of texts using Mistral's embedding API.
    Batches requests to stay within API limits.
    """
    if not texts:
        return []

    sess     = _get_session()
    all_embs: List[List[float]] = []

    for batch in _chunk_iterable(texts, batch_size):
        payload = {"model": EMBED_MODEL, "input": batch}
        last_exc = None

        for attempt in range(3):
            try:
                r = sess.post(
                    MISTRAL_EMBED_URL,
                    json=payload,
                    headers=_auth_headers(),
                    timeout=60,
                )
                r.raise_for_status()
                data = r.json()
                # Mistral returns data sorted by index
                batch_embs = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
                all_embs.extend(batch_embs)
                break

            except requests.HTTPError as he:
                last_exc = he
                code = he.response.status_code
                if code == 429 or 500 <= code < 600:
                    time.sleep(1.5 + attempt)
                    continue
                raise

            except Exception as e:
                last_exc = e
                time.sleep(1 + attempt)
                continue
        else:
            raise RuntimeError(f"Mistral embeddings failed after retries: {last_exc}")

    return all_embs


def embed_query(text: str) -> List[float]:
    """Return embedding for a single query string."""
    if not text:
        return []
    result = generate_embeddings([text])
    return result[0] if result else []


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing Mistral chat...")
    print(chat("Hello! Explain RAG in simple words."))

    print("\nTesting Mistral embeddings...")
    vec = embed_query("machine learning")
    print(f"Embedding dimensions: {len(vec)}")
    print(f"First 5 values: {vec[:5]}")