"""Deterministic lexical retrieval with BM25 scoring."""

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple

from auto_harness.retrieval.schemas import RetrievalChunk


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:/-]*|[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text or ""))]


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b

    def search(self, query: str, chunks: Iterable[RetrievalChunk], limit: int = 30) -> List[Tuple[RetrievalChunk, float]]:
        chunks = list(chunks)
        terms = list(dict.fromkeys(tokenize(query)))
        if not chunks or not terms:
            return []
        tokenized = [tokenize(chunk.text + " " + chunk.path + " " + chunk.symbol) for chunk in chunks]
        avgdl = sum(len(tokens) for tokens in tokenized) / max(1, len(tokenized))
        doc_freq = {term: sum(1 for tokens in tokenized if term in set(tokens)) for term in terms}
        ranked = []
        for chunk, tokens in zip(chunks, tokenized):
            counts = Counter(tokens)
            score = 0.0
            for term in terms:
                tf = counts[term]
                if not tf:
                    continue
                df = doc_freq[term]
                idf = math.log(1.0 + (len(chunks) - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1.0 - self.b + self.b * len(tokens) / max(1.0, avgdl))
                score += idf * (tf * (self.k1 + 1.0)) / denom
            if score > 0:
                ranked.append((chunk, score))
        return sorted(ranked, key=lambda item: (-item[1], item[0].chunk_id))[:limit]

