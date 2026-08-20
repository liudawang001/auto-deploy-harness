"""Transactional SQLite storage for retrieval documents, chunks, and vectors."""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from auto_harness.retrieval.schemas import RetrievalChunk, RetrievalDocument, RetrievalManifest


class RetrievalStore:
    def __init__(self, path: Path, *, fault_hook=None) -> None:
        self.path = Path(path)
        self.fault_hook = fault_hook
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fts5_available = False
        self._initialize()

    def connect(self):
        connection = sqlite3.connect(str(self.path), timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS documents (
                  document_id TEXT PRIMARY KEY, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                  chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
                  source_type TEXT NOT NULL, repository_fingerprint TEXT NOT NULL,
                  task_id TEXT NOT NULL, text_value TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_scope
                  ON chunks(repository_fingerprint, task_id, source_type);
                CREATE TABLE IF NOT EXISTS embeddings (
                  chunk_id TEXT NOT NULL, identity TEXT NOT NULL,
                  vector TEXT NOT NULL, PRIMARY KEY(chunk_id, identity)
                );
                CREATE TABLE IF NOT EXISTS manifests (
                  scope TEXT PRIMARY KEY, payload TEXT NOT NULL
                );
                """
            )
            try:
                connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS lexical_fts USING fts5(chunk_id UNINDEXED, text_value)")
                self.fts5_available = True
            except sqlite3.OperationalError:
                self.fts5_available = False

    def replace(self, documents: Iterable[RetrievalDocument], chunks: Iterable[RetrievalChunk], manifest: RetrievalManifest, embedding_payload=None) -> None:
        documents, chunks = list(documents), list(chunks)
        manifest.finalize()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM documents")
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM embeddings")
            if self.fts5_available:
                connection.execute("DELETE FROM lexical_fts")
            connection.executemany(
                "INSERT INTO documents(document_id,payload) VALUES(?,?)",
                [(item.document_id, json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True)) for item in documents],
            )
            connection.executemany(
                "INSERT INTO chunks(chunk_id,document_id,source_type,repository_fingerprint,task_id,text_value,payload) VALUES(?,?,?,?,?,?,?)",
                [(
                    item.chunk_id, item.document_id, item.source_type,
                    item.repository_fingerprint, item.task_id, item.text,
                    json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True),
                ) for item in chunks],
            )
            self._fault("after_chunk_write_before_lexical_commit")
            if self.fts5_available:
                connection.executemany(
                    "INSERT INTO lexical_fts(chunk_id,text_value) VALUES(?,?)",
                    [(item.chunk_id, item.text) for item in chunks],
                )
            self._fault("after_lexical_write_before_embedding_commit")
            if embedding_payload:
                identity, vectors = embedding_payload
                connection.executemany(
                    "INSERT OR REPLACE INTO embeddings(chunk_id,identity,vector) VALUES(?,?,?)",
                    [(chunk_id, identity, json.dumps(vector, separators=(",", ":"))) for chunk_id, vector in vectors.items()],
                )
            self._fault("after_embedding_write_before_manifest")
            connection.execute(
                "INSERT OR REPLACE INTO manifests(scope,payload) VALUES(?,?)",
                (manifest.repository_fingerprint or "global", json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True)),
            )

    def _fault(self, point: str) -> None:
        if self.fault_hook is not None:
            self.fault_hook(point)

    def manifest(self, scope: str = "global") -> Optional[RetrievalManifest]:
        with self.connect() as connection:
            row = connection.execute("SELECT payload FROM manifests WHERE scope=?", (scope,)).fetchone()
        if row is None:
            return None
        manifest = RetrievalManifest(**json.loads(row["payload"]))
        manifest.validate()
        return manifest

    def chunks(self, query=None) -> List[RetrievalChunk]:
        clauses, params = [], []
        if query is not None:
            placeholders = ",".join("?" for _ in query.sources)
            clauses.append("source_type IN (%s)" % placeholders)
            params.extend(query.sources)
            if query.repository_fingerprint:
                clauses.append("(repository_fingerprint='' OR repository_fingerprint=?)")
                params.append(query.repository_fingerprint)
            if query.task_id:
                clauses.append("(task_id='' OR task_id=?)")
                params.append(query.task_id)
        sql = "SELECT payload FROM chunks"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [RetrievalChunk.from_dict(json.loads(row["payload"])) for row in rows]

    def lexical_search(self, query_text: str, chunks: Iterable[RetrievalChunk], limit: int):
        """Use SQLite FTS5 BM25 over an already policy-filtered candidate set."""
        if not self.fts5_available:
            raise RuntimeError("SQLite FTS5 is unavailable")
        from auto_harness.retrieval.lexical import tokenize
        terms = tokenize(query_text)
        if not terms:
            return []
        expression = " OR ".join('"%s"' % term.replace('"', '""') for term in terms[:32])
        allowed = {item.chunk_id: item for item in chunks}
        if not allowed:
            return []
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT chunk_id,bm25(lexical_fts) AS score FROM lexical_fts "
                "WHERE lexical_fts MATCH ? ORDER BY score,chunk_id LIMIT ?",
                (expression, max(int(limit) * 8, int(limit))),
            ).fetchall()
        result = []
        for row in rows:
            chunk = allowed.get(row["chunk_id"])
            if chunk is not None:
                result.append((chunk, -float(row["score"])))
            if len(result) >= int(limit):
                break
        return result

    def save_embeddings(self, identity: str, vectors: Dict[str, List[float]]) -> None:
        with self.connect() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO embeddings(chunk_id,identity,vector) VALUES(?,?,?)",
                [(chunk_id, identity, json.dumps(vector, separators=(",", ":"))) for chunk_id, vector in vectors.items()],
            )

    def embeddings(self, identity: str, chunk_ids: Optional[Iterable[str]] = None) -> Dict[str, List[float]]:
        params = [identity]
        sql = "SELECT chunk_id,vector FROM embeddings WHERE identity=?"
        ids = list(chunk_ids or [])
        if ids:
            sql += " AND chunk_id IN (%s)" % ",".join("?" for _ in ids)
            params.extend(ids)
        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return {row["chunk_id"]: json.loads(row["vector"]) for row in rows}
