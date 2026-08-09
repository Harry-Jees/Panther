"""Query answering: retrieval, prompt building, token budgeting.

The pipeline: embed the query → retrieve top-k chunks from the vector store →
budget the context into the configured prompt-token allowance → call the LLM
(primary with fallback) → return the answer plus metadata.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional
from pathlib import Path

from . import config as config_mod
from .agent import AgentResult, run_agent, run_agent_stream
from .api import TokenRateLimiter, call_llm, estimate_tokens, stream_chat
from .rag import EmbeddingPipeline, VectorStore

SYSTEM_PROMPT = """You are Panther, a knowledgeable and encouraging coding instructor.

Your job is to teach the user about THEIR codebase — explain how things work,
point out interesting patterns, and make connections. Always:
- Answer in the voice of a supportive teacher, not an exam grader.
- Reference the provided code chunks by file path when you use them.
- Ask a follow-up question to deepen understanding when appropriate.
- If the provided chunks do not answer the question, say so honestly and
  suggest what else to look at — never invent code that is not in the chunks.
- Keep explanations clear and well-structured; use short code examples only
  when they genuinely clarify the idea."""

# Process-wide rate limiter (soft TPM cap per guide).
_RATE_LIMITER = TokenRateLimiter()


class Answer:
    """Result of an answering attempt.

    ``text`` is ``None`` when the full LLM pipeline could not run (missing
    dependencies, missing API keys, or provider failures); ``warning`` then
    explains why so the caller can fall back to keyword retrieval.
    """

    def __init__(
        self,
        text: Optional[str],
        model_used: Optional[str] = None,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
        tokens: Optional[Dict[str, int]] = None,
        warning: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.text = text
        self.model_used = model_used
        self.retrieved_chunks = retrieved_chunks or []
        self.tokens = tokens
        self.warning = warning
        self.extra = extra or {}


def retrieve_context(query: str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Embed the query and return the top-k relevant chunks (sorted by relevance).

    When reranking is enabled (and a cross-encoder is available), the candidate
    list is re-scored against the query for better precision.  Reranking is
    best-effort: if the optional model cannot be loaded, the original ordering
    is used without failing the pipeline.
    """
    retrieval = cfg.get("retrieval", {})
    embedder = EmbeddingPipeline(retrieval.get("embedding_model", "all-MiniLM-L6-v2"))
    store = VectorStore(config_mod.db_path() / "chroma")
    query_embedding = embedder.embed_query(query)
    top_k = retrieval.get("top_k_chunks", 5)
    # Fetch a few extra candidates so reranking has headroom to reorder.
    n_results = top_k * 3 if _rerank_enabled(cfg) else top_k
    results = store.query(query_embedding, n_results=n_results)

    threshold = retrieval.get("similarity_threshold", 0.0)
    ranked: List[Dict[str, Any]] = []
    for result in results:
        # Chroma returns cosine distance; smaller is more similar.
        similarity = max(0.0, 1.0 - float(result.get("distance", 1.0)))
        if similarity >= threshold:
            ranked.append(
                {
                    "text": result.get("text", ""),
                    "file_path": result.get("file_path", "?"),
                    "start_line": result.get("start_line"),
                    "end_line": result.get("end_line"),
                    "similarity": round(similarity, 3),
                }
            )
    ranked.sort(key=lambda c: c["similarity"], reverse=True)
    if _rerank_enabled(cfg) and len(ranked) > 1:
        ranked = _rerank(query, ranked, top_k)
    return ranked[:top_k]


def _rerank_enabled(cfg: Dict[str, Any]) -> bool:
    return bool((cfg.get("retrieval") or {}).get("rerank", False))


_RERANK_MODEL = None


def _rerank(query: str, chunks: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Re-score chunks with a local cross-encoder; fall back to input order on error."""
    global _RERANK_MODEL
    try:
        if _RERANK_MODEL is None:
            from sentence_transformers import CrossEncoder

            _RERANK_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
        texts = [c["text"] for c in chunks]
        scores = _RERANK_MODEL.predict([(query, t) for t in texts])
    except Exception:
        return chunks  # graceful degradation: keep retrieval order
    pairs = sorted(zip(chunks, scores), key=lambda p: float(p[1]), reverse=True)
    return [
        {**chunk, "similarity": round(float(score), 3)}
        for chunk, score in pairs[:top_k]
    ]


def build_messages(
    query: str, chunks: List[Dict[str, Any]], cfg: Dict[str, Any], attachments: Optional[List[Dict[str, str]]] = None
) -> tuple:
    """Build the system+user message list, trimming chunks to the token budget.

    Returns ``(messages, kept_chunks)``.  Chunks are assumed pre-sorted by
    relevance; the least relevant ones are dropped first to stay under
    ``max_prompt_tokens``.  ``attachments`` (from ``@file`` mentions) are
    prepended to the user content and consume budget first, since the user
    explicitly referenced them.
    """
    budget = cfg.get("token_budget", {}).get("max_prompt_tokens", 28000)
    sys_tokens = estimate_tokens(SYSTEM_PROMPT)
    q_tokens = estimate_tokens(query)
    available = budget - sys_tokens - q_tokens

    attach_block = ""
    if attachments:
        parts = []
        for att in attachments:
            parts.append(f"--- {att['path']} ---\n{att['text']}")
        attach_block = "\n\nUser-attached files (explicitly referenced):\n\n" + "\n\n".join(parts)
        available -= estimate_tokens(attach_block)

    kept: List[Dict[str, Any]] = []
    used = 0
    for chunk in chunks:
        cost = estimate_tokens(chunk.get("text", ""))
        if kept and used + cost > available:
            break
        kept.append(chunk)
        used += cost

    if kept:
        context = "\n\n".join(
            f"--- Chunk {i}: {c['file_path']} "
            f"(lines {c['start_line']}-{c['end_line']}) ---\n{c['text']}"
            for i, c in enumerate(kept, 1)
        )
        user_content = f"Question: {query}\n\nRelevant code from the repository:\n\n{context}{attach_block}"
    elif chunks:
        user_content = (
            f"Question: {query}\n\n(Relevant chunks were trimmed to fit the token budget.){attach_block}"
        )
    else:
        user_content = f"Question: {query}\n\n(No relevant code chunks found in the index.){attach_block}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages, kept


def answer_question(
    query: str,
    project_root: Optional[Path] = None,
    cfg: Optional[Dict[str, Any]] = None,
    resolve_key: Optional[Callable[[str], Optional[str]]] = None,
    attachments: Optional[List[Dict[str, str]]] = None,
) -> Answer:
    """Full RAG + LLM answer pipeline with graceful degradation."""
    cfg = cfg or config_mod.load_config()

    try:
        chunks = retrieve_context(query, cfg)
    except Exception as exc:  # missing deps / no index yet
        return Answer(None, warning=_friendly_retrieval_error(exc))

    try:
        messages, kept = build_messages(query, chunks, cfg, attachments=attachments)
    except Exception as exc:
        return Answer(None, warning=f"Could not build the prompt: {exc}")

    try:
        _RATE_LIMITER.wait_if_needed()
        result = call_llm(messages, cfg, resolve_key=resolve_key)
        _RATE_LIMITER.add(result.get("prompt_tokens", 0) + result.get("completion_tokens", 0))
    except Exception as exc:
        return Answer(None, warning=str(exc))

    tokens = {
        "prompt": result.get("prompt_tokens", 0)
        or estimate_tokens(messages[1]["content"]),
        "completion": result.get("completion_tokens", 0),
    }
    return Answer(
        text=result["content"],
        model_used=result.get("model"),
        retrieved_chunks=kept,
        tokens=tokens,
    )


def agent_answer(
    query: str,
    project_root: Optional[Path] = None,
    cfg: Optional[Dict[str, Any]] = None,
    resolve_key: Optional[Callable[[str], Optional[str]]] = None,
    on_tool: Optional[Callable] = None,
    on_tool_start: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    attachments: Optional[List[Dict[str, str]]] = None,
) -> Answer:
    """RAG + tool-calling agent answer with graceful degradation.

    Returns an :class:`Answer` where ``text`` is ``None`` when the pipeline
    could not run; tool events are attached to ``answer.extra["tool_events"]``.
    """
    cfg = cfg or config_mod.load_config()
    root = project_root or Path.cwd()

    try:
        chunks = retrieve_context(query, cfg)
    except Exception as exc:  # missing deps / no index yet
        return Answer(None, warning=_friendly_retrieval_error(exc))

    try:
        messages, kept = build_messages(query, chunks, cfg, attachments=attachments)
    except Exception as exc:
        return Answer(None, warning=f"Could not build the prompt: {exc}")

    try:
        _RATE_LIMITER.wait_if_needed()
        max_turns = (cfg.get("tools") or {}).get("max_turns", 6)
        result: AgentResult = run_agent(
            messages,
            root,
            cfg,
            resolve_key=resolve_key,
            max_turns=max_turns,
            on_tool=on_tool,
            on_tool_start=on_tool_start,
        )
        _RATE_LIMITER.add(result.prompt_tokens + result.completion_tokens)
    except Exception as exc:
        return Answer(None, warning=str(exc))

    if result.content is None:
        # Keep the agent's real accounting so the CLI fallback can still
        # report accurate token/model stats after a budget exhaustion.
        answer = Answer(None, warning=result.warning or "The agent produced no answer.")
        answer.extra = {
            "tool_events": result.tool_events,
            "tokens": {
                "prompt": result.prompt_tokens or 0,
                "completion": result.completion_tokens or 0,
            },
            "model_used": result.model,
        }
        return answer

    tokens = {
        "prompt": result.prompt_tokens or estimate_tokens(messages[1]["content"]),
        "completion": result.completion_tokens,
    }
    answer = Answer(
        text=result.content,
        model_used=result.model,
        retrieved_chunks=kept,
        tokens=tokens,
    )
    answer.extra = {"tool_events": result.tool_events}
    return answer


def stream_answer(
    query: str,
    project_root: Optional[Path] = None,
    cfg: Optional[Dict[str, Any]] = None,
    resolve_key: Optional[Callable[[str], Optional[str]]] = None,
    attachments: Optional[List[Dict[str, str]]] = None,
) -> Iterator[tuple]:
    """Streamed variant of :func:`answer_question`.

    Yields ``("delta", text)`` tuples as the model responds, then a final
    ``("done", Answer)`` event.  ``Answer.text`` is ``None`` when the pipeline
    could not run (warning explains why).
    """
    cfg = cfg or config_mod.load_config()

    try:
        chunks = retrieve_context(query, cfg)
    except Exception as exc:  # missing deps / no index yet
        yield ("done", Answer(None, warning=_friendly_retrieval_error(exc)))
        return

    try:
        messages, kept = build_messages(query, chunks, cfg, attachments=attachments)
    except Exception as exc:
        yield ("done", Answer(None, warning=f"Could not build the prompt: {exc}"))
        return

    try:
        _RATE_LIMITER.wait_if_needed()
        text = ""
        for kind, payload in stream_chat(messages, cfg, resolve_key=resolve_key):
            if kind == "delta":
                text += payload
                yield ("delta", payload)
            else:  # "done"
                _RATE_LIMITER.add(
                    payload.get("prompt_tokens", 0) + payload.get("completion_tokens", 0)
                )
                tokens = {
                    "prompt": payload.get("prompt_tokens", 0)
                    or estimate_tokens(messages[1]["content"]),
                    "completion": payload.get("completion_tokens", 0),
                }
                yield (
                    "done",
                    Answer(
                        text=text,
                        model_used=payload.get("model"),
                        retrieved_chunks=kept,
                        tokens=tokens,
                    ),
                )
                return
    except Exception as exc:
        yield ("done", Answer(None, warning=str(exc)))


def stream_agent_answer(
    query: str,
    project_root: Optional[Path] = None,
    cfg: Optional[Dict[str, Any]] = None,
    resolve_key: Optional[Callable[[str], Optional[str]]] = None,
    attachments: Optional[List[Dict[str, str]]] = None,
    on_tool: Optional[Callable] = None,
    on_tool_start: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Iterator[tuple]:
    """RAG + tool-calling agent answer with live streaming.

    Yields ``("delta", text)`` chunks as the final answer streams in, then a
    final ``("done", Answer)`` event.  Tool executions surface through
    ``on_tool_start`` / ``on_tool`` callbacks (the same contract as
    :func:`agent_answer`) so the UI can animate them while the stream runs.
    """
    cfg = cfg or config_mod.load_config()
    root = project_root or Path.cwd()

    try:
        chunks = retrieve_context(query, cfg)
    except Exception as exc:  # missing deps / no index yet
        yield ("done", Answer(None, warning=_friendly_retrieval_error(exc)))
        return

    try:
        messages, kept = build_messages(query, chunks, cfg, attachments=attachments)
    except Exception as exc:
        yield ("done", Answer(None, warning=f"Could not build the prompt: {exc}"))
        return

    try:
        _RATE_LIMITER.wait_if_needed()
        max_turns = (cfg.get("tools") or {}).get("max_turns", 6)
        result: Optional[AgentResult] = None
        for kind, payload in run_agent_stream(
            messages,
            root,
            cfg,
            resolve_key=resolve_key,
            max_turns=max_turns,
            on_tool=on_tool,
            on_tool_start=on_tool_start,
        ):
            if kind == "delta":
                yield ("delta", payload)
            elif kind == "tool":
                continue  # surfaced through callbacks; collected in extra below
            elif kind == "done":
                result = payload
        if result is None:
            yield ("done", Answer(None, warning="The agent produced no response."))
            return
        _RATE_LIMITER.add(result.prompt_tokens + result.completion_tokens)
    except Exception as exc:
        yield ("done", Answer(None, warning=str(exc)))
        return

    if result.content is None:
        answer = Answer(None, warning=result.warning or "The agent produced no answer.")
        answer.extra = {
            "tool_events": result.tool_events,
            "tokens": {
                "prompt": result.prompt_tokens or 0,
                "completion": result.completion_tokens or 0,
            },
            "model_used": result.model,
        }
        yield ("done", answer)
        return

    answer = Answer(
        text=result.content,
        model_used=result.model,
        retrieved_chunks=kept,
        tokens={
            "prompt": result.prompt_tokens or estimate_tokens(messages[1]["content"]),
            "completion": result.completion_tokens,
        },
    )
    answer.extra = {"tool_events": result.tool_events}
    yield ("done", answer)


def _friendly_retrieval_error(exc: Exception) -> str:
    message = str(exc)
    if "sentence-transformers" in message:
        return (
            "Vector retrieval needs sentence-transformers. Run "
            "`pip install sentence-transformers chromadb`, then `panther index`."
        )
    if "chromadb" in message:
        return "Vector retrieval needs chromadb. Run `pip install chromadb`, then `panther index`."
    if "greater than number of elements" in message or "No such collection" in message or "does not exist" in message:
        return "This project's index is empty or smaller than requested. Run `panther index` first."
    return f"Retrieval failed: {message}"
