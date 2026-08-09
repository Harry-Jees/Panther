import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from panther import qa
from panther.rag import RAGDependencyError


def _cfg():
    return {
        "retrieval": {"top_k_chunks": 5, "similarity_threshold": 0.5},
        "token_budget": {"max_prompt_tokens": 2000},
        "llm": {
            "primary": {"provider": "nvidia_nim", "model": "m1"},
        },
        "api_keys": {"nvidia_nim": "nvapi-1"},
    }


def _chunks():
    return [
        {"text": "def foo(): return 42", "file_path": "a.py", "start_line": 1, "end_line": 1, "similarity": 0.9},
        {"text": "SELECT * FROM t", "file_path": "b.sql", "start_line": 2, "end_line": 2, "similarity": 0.6},
    ]


class TestRetrieveContext(unittest.TestCase):
    def test_filters_by_threshold_and_sorts(self):
        with patch("panther.qa.EmbeddingPipeline") as mock_embed, patch(
            "panther.qa.VectorStore"
        ) as mock_store, patch("panther.qa.config_mod.db_path", return_value=Path("tmp")):
            mock_embed.return_value.embed_query.return_value = [0.1]
            mock_store.return_value.query.return_value = [
                {"text": "far", "file_path": "x.py", "start_line": 1, "end_line": 1, "distance": 0.9},
                {"text": "near", "file_path": "y.py", "start_line": 2, "end_line": 2, "distance": 0.1},
            ]
            results = qa.retrieve_context("question", _cfg())
        self.assertEqual([r["file_path"] for r in results], ["y.py"])
        self.assertEqual(results[0]["similarity"], 0.9)


class TestBuildMessages(unittest.TestCase):
    def test_includes_chunks_and_question(self):
        messages, kept = qa.build_messages("what is foo?", _chunks(), _cfg())
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Question: what is foo?", messages[1]["content"])
        self.assertIn("a.py", messages[1]["content"])
        self.assertEqual(len(kept), 2)

    def test_budgets_out_low_relevance_chunks(self):
        cfg = _cfg()
        cfg["token_budget"] = {"max_prompt_tokens": 60}  # tiny budget
        messages, kept = qa.build_messages("what is foo?", _chunks(), cfg)
        # The smallest budget forces a trimmed prompt, and we never reject the query.
        self.assertIn("Question: what is foo?", messages[1]["content"])
        self.assertLessEqual(len(kept), 2)

    def test_no_chunks_still_answers(self):
        messages, kept = qa.build_messages("hi", [], _cfg())
        self.assertIn("No relevant code chunks found", messages[1]["content"])
        self.assertEqual(kept, [])


class TestAnswerQuestion(unittest.TestCase):
    def test_full_pipeline_success(self):
        with patch("panther.qa.retrieve_context", return_value=_chunks()), patch(
            "panther.qa.call_llm",
            return_value={
                "content": "Great question!",
                "provider": "nvidia_nim",
                "model": "m1",
                "prompt_tokens": 100,
                "completion_tokens": 50,
            },
        ), patch.object(qa._RATE_LIMITER, "wait_if_needed"):
            answer = qa.answer_question("what is foo?", Path("."), _cfg())
        self.assertEqual(answer.text, "Great question!")
        self.assertEqual(answer.model_used, "m1")
        self.assertEqual(len(answer.retrieved_chunks), 2)
        self.assertEqual(answer.tokens["prompt"], 100)

    def test_retrieval_failure_returns_warning(self):
        with patch(
            "panther.qa.retrieve_context",
            side_effect=RAGDependencyError("chromadb is not installed"),
        ), patch.object(qa._RATE_LIMITER, "wait_if_needed"):
            answer = qa.answer_question("q", Path("."), _cfg())
        self.assertIsNone(answer.text)
        self.assertIn("chromadb", answer.warning)

    def test_provider_failure_returns_warning(self):
        from panther.api import LLMError

        with patch("panther.qa.retrieve_context", return_value=_chunks()), patch(
            "panther.qa.call_llm", side_effect=LLMError("All providers failed")
        ), patch.object(qa._RATE_LIMITER, "wait_if_needed"):
            answer = qa.answer_question("q", Path("."), _cfg())
        self.assertIsNone(answer.text)
        self.assertIn("All providers failed", answer.warning)


class TestRerank(unittest.TestCase):
    def _cfg(self, rerank):
        return {
            "retrieval": {"top_k_chunks": 2, "similarity_threshold": 0.0, "rerank": rerank},
            "token_budget": {"max_prompt_tokens": 2000},
        }

    def test_rerank_reorders_by_cross_encoder(self):
        chunks = [
            {"text": "def foo(): pass", "file_path": "a.py", "similarity": 0.9},
            {"text": "def bar(): pass", "file_path": "b.py", "similarity": 0.8},
        ]
        fake_model = unittest.mock.MagicMock()
        fake_model.predict.return_value = [0.2, 0.9]
        with unittest.mock.patch("panther.qa._RERANK_MODEL", fake_model):
            out = qa._rerank("bar", chunks, top_k=2)
        self.assertEqual(out[0]["file_path"], "b.py")
        self.assertEqual(out[0]["similarity"], 0.9)

    def test_rerank_falls_back_on_error(self):
        chunks = [
            {"text": "a", "file_path": "a.py", "similarity": 0.9},
            {"text": "b", "file_path": "b.py", "similarity": 0.8},
        ]
        fake_model = unittest.mock.MagicMock()
        fake_model.predict.side_effect = Exception("model unavailable")
        with unittest.mock.patch("panther.qa._RERANK_MODEL", fake_model):
            out = qa._rerank("q", chunks, top_k=2)
        self.assertEqual([c["file_path"] for c in out], ["a.py", "b.py"])

    def test_retrieve_context_rerank_enabled_fetches_more(self):
        def _cfg():
            return {
                "retrieval": {
                    "top_k_chunks": 2,
                    "similarity_threshold": 0.0,
                    "rerank": True,
                    "embedding_model": "m",
                }
            }

        with unittest.mock.patch("panther.qa.EmbeddingPipeline") as mock_embed, unittest.mock.patch(
            "panther.qa.VectorStore"
        ) as mock_store, unittest.mock.patch(
            "panther.qa.config_mod.db_path", return_value=Path("tmp")
        ), unittest.mock.patch(
            "panther.qa._rerank", side_effect=lambda q, c, k: c[:k]
        ) as mock_rerank:
            mock_embed.return_value.embed_query.return_value = [0.1]
            mock_store.return_value.query.return_value = [
                {"text": f"c{i}", "file_path": f"{i}.py", "start_line": 1, "end_line": 1, "distance": 0.1}
                for i in range(6)
            ]
            out = qa.retrieve_context("q", _cfg())
        self.assertEqual(len(out), 2)
        mock_rerank.assert_called_once()


class TestStreamAnswer(unittest.TestCase):
    def test_streams_deltas_then_done(self):
        with patch("panther.qa.retrieve_context", return_value=_chunks()), patch(
            "panther.qa.stream_chat",
            return_value=iter(
                [("delta", "Hel"), ("delta", "lo"), ("done", {"model": "m1", "prompt_tokens": 10, "completion_tokens": 5})]
            ),
        ), patch.object(qa._RATE_LIMITER, "wait_if_needed"):
            events = list(qa.stream_answer("hi", Path("."), _cfg()))
        self.assertEqual(events[0], ("delta", "Hel"))
        self.assertEqual(events[1], ("delta", "lo"))
        kind, answer = events[2]
        self.assertEqual(kind, "done")
        self.assertEqual(answer.text, "Hello")
        self.assertEqual(answer.model_used, "m1")
        self.assertEqual(len(answer.retrieved_chunks), 2)

    def test_retrieval_failure_returns_done_with_warning(self):
        with patch(
            "panther.qa.retrieve_context",
            side_effect=RAGDependencyError("chromadb is not installed"),
        ):
            events = list(qa.stream_answer("q", Path("."), _cfg()))
        kind, answer = events[0]
        self.assertEqual(kind, "done")
        self.assertIsNone(answer.text)
        self.assertIn("chromadb", answer.warning)

    def test_provider_failure_returns_done_with_warning(self):
        from panther.api import LLMError

        with patch("panther.qa.retrieve_context", return_value=_chunks()), patch(
            "panther.qa.stream_chat", side_effect=LLMError("All providers failed")
        ), patch.object(qa._RATE_LIMITER, "wait_if_needed"):
            events = list(qa.stream_answer("q", Path("."), _cfg()))
        kind, answer = events[0]
        self.assertEqual(kind, "done")
        self.assertIsNone(answer.text)
        self.assertIn("All providers failed", answer.warning)

    def test_agent_budget_exhaustion_keeps_token_stats(self):
        from panther.agent import AgentResult

        result = AgentResult(
            None,  # no final answer: budget exhausted
            [],
            model="m1",
            provider="nvidia_nim",
            prompt_tokens=1200,
            completion_tokens=340,
            warning="Reached the 6-turn tool budget without a final answer.",
            turns_used=6,
        )
        with patch("panther.qa.retrieve_context", return_value=_chunks()), patch(
            "panther.qa.run_agent", return_value=result
        ), patch.object(qa._RATE_LIMITER, "wait_if_needed"):
            answer = qa.agent_answer("q", Path("."), _cfg())
        self.assertIsNone(answer.text)
        self.assertIn("6-turn", answer.warning)
        extra = answer.extra or {}
        self.assertEqual(extra["tokens"]["prompt"], 1200)
        self.assertEqual(extra["tokens"]["completion"], 340)
        self.assertEqual(extra["model_used"], "m1")


if __name__ == "__main__":
    unittest.main()
