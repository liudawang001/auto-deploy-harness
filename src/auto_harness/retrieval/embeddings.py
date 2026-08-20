"""Optional embedding providers. External transport is explicit and bounded."""

import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
import uuid
from typing import List

from auto_harness.retrieval.lexical import tokenize
from auto_harness.retrieval.schemas import EmbeddingResult


def _normalize(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


class FakeEmbeddingProvider:
    """Deterministic offline provider for contracts and CI, never live evidence."""
    name = "fake"
    model = "hashing-v1"

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def _vector(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            slot = int.from_bytes(digest[:4], "big") % self.dimension
            vector[slot] += -1.0 if digest[4] & 1 else 1.0
        return _normalize(vector)

    def _result(self, texts: List[str]) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=[self._vector(text) for text in texts], provider=self.name,
            model=self.model, dimension=self.dimension,
            request_id="fake_" + uuid.uuid4().hex, input_count=len(texts),
        )

    def embed_documents(self, texts: List[str], *, request_context=None) -> EmbeddingResult:
        return self._result(list(texts))

    def embed_query(self, text: str, *, request_context=None) -> EmbeddingResult:
        return self._result([text])


class OpenAICompatibleEmbeddingProvider:
    name = "openai_compatible"

    def __init__(self, *, api_base: str, model: str, api_key_env: str, dimension: int, timeout_seconds: int = 30, batch_size: int = 64) -> None:
        if not api_base.startswith("https://"):
            raise ValueError("embedding api_base must use https")
        if not model or not api_key_env or dimension < 1:
            raise ValueError("embedding model, key env, and dimension are required")
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.dimension = dimension
        self.timeout_seconds = timeout_seconds
        self.batch_size = max(1, min(int(batch_size), 256))

    def embed_documents(self, texts: List[str], *, request_context=None) -> EmbeddingResult:
        texts = list(texts)
        vectors, request_ids, usage, latency = [], [], {}, 0
        for offset in range(0, len(texts), self.batch_size):
            result = self._embed(texts[offset: offset + self.batch_size])
            vectors.extend(result.vectors)
            if result.request_id:
                request_ids.append(result.request_id)
            latency += result.latency_ms
            for key, value in result.token_usage.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage[key] = usage.get(key, 0) + value
        return EmbeddingResult(
            vectors=vectors, provider=self.name, model=self.model,
            dimension=self.dimension, request_id=",".join(request_ids)[:500],
            input_count=len(texts), token_usage=usage, latency_ms=latency,
        )

    def embed_query(self, text: str, *, request_context=None) -> EmbeddingResult:
        return self._embed([text])

    def _embed(self, texts: List[str]) -> EmbeddingResult:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise RuntimeError("embedding API key environment variable is unavailable: %s" % self.api_key_env)
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            self.api_base + "/embeddings", data=payload, method="POST",
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
                request_id = str(response.headers.get("x-request-id", ""))
        except (OSError, ValueError, urllib.error.HTTPError) as exc:
            raise RuntimeError("embedding provider request failed: %s" % type(exc).__name__) from exc
        ordered = sorted(body.get("data") or [], key=lambda item: int(item.get("index", 0)))
        vectors = [_normalize([float(value) for value in item.get("embedding", [])]) for item in ordered]
        return EmbeddingResult(
            vectors=vectors, provider=self.name, model=self.model,
            dimension=self.dimension, request_id=request_id,
            input_count=len(texts), token_usage=dict(body.get("usage") or {}),
            latency_ms=int((time.monotonic() - started) * 1000),
        )


def embedding_identity(provider) -> str:
    return "%s:%s:%s:l2" % (provider.name, provider.model, provider.dimension)
