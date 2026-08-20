"""Deterministic, bounded chunkers for repository and memory evidence."""

import ast
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from auto_harness.retrieval.schemas import RetrievalChunk, RetrievalDocument


def estimate_tokens(text: str) -> int:
    return max(1, (len(str(text).encode("utf-8")) + 3) // 4)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_chunk(document: RetrievalDocument, ordinal: int, text: str, start: int, end: int, version: str, symbol: str = "") -> RetrievalChunk:
    digest = _sha(text)
    identity = "|".join((document.source_identity, document.content_sha256, version, str(start), str(end), symbol, digest))
    chunk_id = "chk_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return RetrievalChunk(
        chunk_id=chunk_id, document_id=document.document_id,
        chunker_version=version, ordinal=ordinal, text=text,
        text_sha256=digest, token_estimate=estimate_tokens(text),
        path=document.path, symbol=symbol, line_start=start, line_end=end,
        source_type=document.source_type,
        repository_fingerprint=document.repository_fingerprint,
        task_id=document.task_id, trust_level=document.trust_level,
        metadata={
            **document.metadata,
            "stage": list(document.stage_tags),
            "frameworks": list(document.framework_tags),
        },
    )


class LineWindowChunker:
    version = "line_window_v1"

    def __init__(self, max_tokens: int = 640, overlap_lines: int = 8) -> None:
        self.max_tokens = max_tokens
        self.overlap_lines = overlap_lines

    def chunk(self, document: RetrievalDocument, text: str) -> List[RetrievalChunk]:
        lines = text.splitlines() or [""]
        result, start, ordinal = [], 0, 0
        while start < len(lines):
            end = start
            selected = []
            while end < len(lines):
                candidate = "\n".join(selected + [lines[end]])
                if selected and estimate_tokens(candidate) > self.max_tokens:
                    break
                selected.append(lines[end])
                end += 1
            value = "\n".join(selected)
            result.append(_make_chunk(document, ordinal, value, start + 1, max(start + 1, end), self.version))
            ordinal += 1
            if end >= len(lines):
                break
            start = max(start + 1, end - self.overlap_lines)
        return result


class PythonAstChunker:
    version = "python_ast_v1"

    def __init__(self, fallback=None) -> None:
        self.fallback = fallback or LineWindowChunker()

    def chunk(self, document: RetrievalDocument, text: str) -> List[RetrievalChunk]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return self.fallback.chunk(document, text)
        lines = text.splitlines()
        nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        if not nodes:
            return self.fallback.chunk(document, text)
        result = []
        prefix_end = max(0, min(node.lineno for node in nodes) - 1)
        if prefix_end:
            result.append(_make_chunk(document, len(result), "\n".join(lines[:prefix_end]), 1, prefix_end, self.version, "<module>"))
        for node in nodes:
            start = int(node.lineno)
            end = int(getattr(node, "end_lineno", node.lineno))
            result.append(_make_chunk(document, len(result), "\n".join(lines[start - 1:end]), start, end, self.version, str(node.name)))
        return result


class MarkdownHeadingChunker:
    version = "markdown_heading_v1"

    def chunk(self, document: RetrievalDocument, text: str) -> List[RetrievalChunk]:
        lines = text.splitlines() or [""]
        starts = [index for index, line in enumerate(lines) if line.lstrip().startswith("#")]
        if not starts:
            return LineWindowChunker().chunk(document, text)
        if starts[0] != 0:
            starts.insert(0, 0)
        result = []
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else len(lines)
            block = "\n".join(lines[start:end])
            symbol = lines[start].strip().lstrip("#").strip() if lines[start].lstrip().startswith("#") else "<preamble>"
            if estimate_tokens(block) > 640:
                for chunk in LineWindowChunker().chunk(document, block):
                    chunk.ordinal = len(result)
                    chunk.line_start += start
                    chunk.line_end += start
                    result.append(chunk)
            else:
                result.append(_make_chunk(document, len(result), block, start + 1, max(start + 1, end), self.version, symbol))
        return result


class StructuredConfigChunker:
    version = "structured_config_v1"

    def chunk(self, document: RetrievalDocument, text: str) -> List[RetrievalChunk]:
        if document.path.lower().endswith(".json"):
            try:
                value = json.loads(text)
            except (TypeError, ValueError):
                return LineWindowChunker().chunk(document, text)
            if isinstance(value, dict):
                return [
                    _make_chunk(document, index, json.dumps({key: item}, ensure_ascii=False, indent=2), 1, max(1, len(text.splitlines())), self.version, str(key))
                    for index, (key, item) in enumerate(value.items())
                ] or LineWindowChunker().chunk(document, text)
        return LineWindowChunker().chunk(document, text)


class ChunkerRegistry:
    def __init__(self) -> None:
        self.python = PythonAstChunker()
        self.markdown = MarkdownHeadingChunker()
        self.structured = StructuredConfigChunker()
        self.fallback = LineWindowChunker()

    def chunk(self, document: RetrievalDocument, text: str) -> List[RetrievalChunk]:
        suffix = Path(document.path).suffix.lower()
        if suffix == ".py":
            return self.python.chunk(document, text)
        if suffix in {".md", ".markdown"} or document.source_type == "active_skill":
            return self.markdown.chunk(document, text)
        if suffix in {".json", ".yaml", ".yml", ".toml"}:
            return self.structured.chunk(document, text)
        if document.source_type in {"issue_memory", "verified_memory"}:
            return [_make_chunk(document, 0, text, 1, max(1, len(text.splitlines())), "memory_entry_v1")]
        return self.fallback.chunk(document, text)

