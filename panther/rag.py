"""RAG indexing engine: repository scanning, chunking, embeddings, vector store.

Heavy dependencies (``sentence-transformers``, ``chromadb``) are imported lazily
so the rest of Panther keeps working with a clear install hint when they are
missing.
"""

from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import config

# Token-based chunking: ~4 characters per token (heuristic estimate).
CHAR_PER_TOKEN = 4

# Process-wide model cache: loading sentence-transformers weights from disk
# takes ~1-2s, so reuse the model across calls within a process (the REPL asks
# many questions per run).
_MODEL_CACHE: Dict[str, Any] = {}

# Model loading prints noisy diagnostics to the console ("sending
# unauthenticated requests", the weight-mismatch table, load progress bars).
# Silence the loggers that emit them so a user-facing REPL stays clean.
_NOISY_LOGGERS = (
    "huggingface_hub",
    "huggingface_hub.utils._http",
    "transformers",
    "sentence_transformers",
    "tokenizers",
    "torch",
)
for _name in _NOISY_LOGGERS:
    logging.getLogger(_name).setLevel(logging.ERROR)


class RAGDependencyError(Exception):
    """Raised when an optional heavy dependency (embeddings / vector store) is missing."""


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


class RepoScanner:
    """Recursive filesystem scanner honouring ignore patterns and extensions."""

    def __init__(
        self,
        ignore_patterns: Optional[List[str]] = None,
        file_extensions: Optional[List[str]] = None,
    ) -> None:
        self.ignore_patterns = set(
            ignore_patterns or config.DEFAULT_CONFIG["indexing"]["ignore_patterns"]
        )
        self.file_extensions = set(
            file_extensions or config.DEFAULT_CONFIG["indexing"]["file_extensions"]
        )

    def scan(self, project_root: Path) -> List[Path]:
        files: List[Path] = []
        for path in sorted(project_root.rglob("*")):
            if path.is_dir():
                continue
            if any(part in self.ignore_patterns for part in path.parts):
                continue
            if path.suffix.lower() not in self.file_extensions:
                continue
            files.append(path)
        return files


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


class CodeChunker:
    """Recursive text splitter respecting paragraph and line boundaries.

    ``chunk_size`` / ``chunk_overlap`` are expressed in tokens (using the
    heuristic ~4 chars per token) to match the guide's 500-token / 50-overlap
    defaults without requiring a tokenizer dependency.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        self.chunk_size = max(1, chunk_size * CHAR_PER_TOKEN)
        self.chunk_overlap = max(0, chunk_overlap * CHAR_PER_TOKEN)

    def split_text(self, text: str) -> List[str]:
        text = text.replace("\r\n", "\n").strip()
        if not text:
            return []
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        for para in paragraphs:
            para_len = len(para)
            if current and current_len + para_len > self.chunk_size:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            if para_len <= self.chunk_size:
                current.append(para)
                current_len += para_len
            else:
                # Paragraph longer than the chunk limit: split by lines.
                chunks.extend(self._split_long_text(para))
        if current:
            chunks.append("\n\n".join(current))
        return self._apply_overlap(chunks)

    def _split_long_text(self, text: str) -> List[str]:
        pieces: List[str] = []
        buf = ""
        for line in text.split("\n"):
            if buf and len(buf) + len(line) + 1 > self.chunk_size:
                pieces.append(buf)
                buf = line
            else:
                buf = f"{buf}\n{line}" if buf else line
        if buf:
            pieces.append(buf)
        return pieces

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks
        result: List[str] = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                tail = chunks[i - 1][-self.chunk_overlap:]
                chunk = f"{tail}\n{chunk}"
            result.append(chunk)
        return result


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class EmbeddingPipeline:
    """Thin wrapper around ``sentence-transformers`` (lazy import)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name

    def _ensure_model(self):
        if self.model_name in _MODEL_CACHE:
            return _MODEL_CACHE[self.model_name]
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RAGDependencyError(
                "sentence-transformers is not installed. Run: "
                "pip install sentence-transformers"
            ) from exc
        # sentence_transformers reconfigures logging when imported (its
        # install_logger bumps verbosity back to WARNING), so re-silence the
        # chatty loggers now that it is in scope.
        for _name in _NOISY_LOGGERS:
            logging.getLogger(_name).setLevel(logging.ERROR)
        # Weight-loading chatter (the transformers mismatch table, tqdm
        # progress bars) writes straight to stdout/stderr; swallow it so a
        # user-facing REPL stays clean. Exceptions still propagate and their
        # tracebacks print after the streams are restored.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            _MODEL_CACHE[self.model_name] = SentenceTransformer(self.model_name)
        return _MODEL_CACHE[self.model_name]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        model = self._ensure_model()
        return model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


class VectorStore:
    """Persistent Chroma-backed vector store for code chunks (lazy import)."""

    def __init__(self, persist_dir: Path, collection_name: str = "code_chunks") -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    def _ensure(self):
        if self._client is not None:
            return
        try:
            import chromadb
        except ImportError as exc:
            raise RAGDependencyError(
                "chromadb is not installed. Run: pip install chromadb"
            ) from exc
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        self._ensure()
        self._collection.add(
            ids=[c["id"] for c in chunks],
            embeddings=[c["embedding"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[
                {
                    "file_path": c["path"],
                    "start_line": c["start_line"],
                    "end_line": c["end_line"],
                }
                for c in chunks
            ],
        )

    def query(self, query_embedding: List[float], n_results: int = 5) -> List[Dict[str, Any]]:
        self._ensure()
        # Chroma raises if n_results exceeds the collection size, so clamp it.
        n_results = max(1, min(n_results, self.count()))
        results = self._collection.query(
            query_embeddings=[query_embedding], n_results=n_results
        )
        out: List[Dict[str, Any]] = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] or {}
            out.append(
                {
                    "text": doc,
                    "file_path": meta.get("file_path"),
                    "start_line": meta.get("start_line"),
                    "end_line": meta.get("end_line"),
                    "distance": results["distances"][0][i],
                }
            )
        return out

    def count(self) -> int:
        self._ensure()
        return self._collection.count()

    def clear(self) -> None:
        self._ensure()
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine"}
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _line_range(content: str, part: str) -> Tuple[int, int]:
    """Approximate the line range of ``part`` within ``content``."""
    idx = content.find(part)
    if idx < 0:
        return (1, 1)
    start = content.count("\n", 0, idx) + 1
    return (start, start + part.count("\n"))


def index_repository(
    project_root: Path,
    cfg: Optional[Dict[str, Any]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> int:
    """Chunk + embed + store the repository. Returns the number of chunks stored.

    Raises :class:`RAGDependencyError` when sentence-transformers or chromadb
    are not installed.
    """
    cfg = cfg or config.load_config()
    retrieval = cfg.get("retrieval", {})

    scanner = RepoScanner(
        ignore_patterns=cfg.get("indexing", {}).get("ignore_patterns"),
        file_extensions=cfg.get("indexing", {}).get("file_extensions"),
    )
    files = scanner.scan(project_root)
    chunker = CodeChunker(
        retrieval.get("chunk_size", 500),
        retrieval.get("chunk_overlap", 50),
    )

    pending: List[Dict[str, Any]] = []
    for path in files:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = path.relative_to(project_root).as_posix()
        for part in chunker.split_text(content):
            start, end = _line_range(content, part)
            pending.append(
                {
                    "id": f"{rel}::{len(pending)}",
                    "text": part,
                    "path": rel,
                    "start_line": start,
                    "end_line": end,
                }
            )

    if not pending:
        return 0

    embedder = EmbeddingPipeline(retrieval.get("embedding_model", "all-MiniLM-L6-v2"))

    # Embed in batches so progress can be reported on large repositories.
    batch_size = 64
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        embeddings = embedder.embed_documents([c["text"] for c in batch])
        for chunk, embedding in zip(batch, embeddings):
            chunk["embedding"] = embedding
        if progress:
            progress(min(len(pending), offset + batch_size), len(pending))

    store = VectorStore(config.db_path() / "chroma")
    store.clear()
    store.add_chunks(pending)
    return len(pending)
