import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from panther import rag


def _make_repo():
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    (root / "a.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    (root / "b.sql").write_text("SELECT * FROM t;\n", encoding="utf-8")
    (root / "c.txt").write_text("just text\n", encoding="utf-8")
    (root / "skip.js").write_text("console.log(1);\n", encoding="utf-8")
    git = root / ".git"
    git.mkdir()
    (git / "config").write_text("[core]", encoding="utf-8")
    nm = root / "node_modules"
    nm.mkdir()
    (nm / "x.js").write_text("ignore", encoding="utf-8")
    return tmp, root


class TestRepoScanner(unittest.TestCase):
    def test_scan_honors_ignores_and_extensions(self):
        tmp, root = _make_repo()
        try:
            files = {p.relative_to(root).as_posix() for p in rag.RepoScanner().scan(root)}
            self.assertIn("a.py", files)
            self.assertIn("b.sql", files)
            self.assertIn("c.txt", files)
            self.assertNotIn("skip.js", files)
            self.assertNotIn(".git/config", files)
            self.assertNotIn("node_modules/x.js", files)
        finally:
            tmp.cleanup()

    def test_custom_extensions(self):
        tmp, root = _make_repo()
        try:
            scanner = rag.RepoScanner(file_extensions=[".py"])
            files = {p.relative_to(root).as_posix() for p in scanner.scan(root)}
            self.assertEqual(files, {"a.py"})
        finally:
            tmp.cleanup()


class TestCodeChunker(unittest.TestCase):
    def test_split_and_overlap(self):
        chunker = rag.CodeChunker(chunk_size=50, chunk_overlap=10)
        text = "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(3))
        chunks = chunker.split_text(text)
        self.assertGreaterEqual(len(chunks), 2)
        # Overlap should carry the tail of the previous chunk forward.
        self.assertIn(chunks[0][-10:].strip(), chunks[1])

    def test_long_paragraph_split_by_lines(self):
        chunker = rag.CodeChunker(chunk_size=20, chunk_overlap=0)
        long_text = "\n".join(f"line {i} with some content" for i in range(50))
        chunks = chunker.split_text(long_text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 20 * 4 + 200 for c in chunks))

    def test_empty_text(self):
        self.assertEqual(rag.CodeChunker().split_text("   \n  "), [])

    def test_line_range(self):
        content = "a\nb\nc\nd\ne"
        start, end = rag._line_range(content, "d\ne")
        self.assertEqual((start, end), (4, 5))


class TestEmbeddingPipeline(unittest.TestCase):
    def tearDown(self):
        rag._MODEL_CACHE.clear()

    def test_models_cached_per_process(self):
        fake_st = MagicMock()
        fake_model = MagicMock()
        fake_st.SentenceTransformer.return_value = fake_model
        with patch.dict("sys.modules", {"sentence_transformers": fake_st}):
            rag.EmbeddingPipeline("cache-test-model").embed_documents(["hi"])
            rag.EmbeddingPipeline("cache-test-model").embed_documents(["bye"])
        # The model weights should be loaded exactly once per model name.
        self.assertEqual(fake_st.SentenceTransformer.call_count, 1)

    def test_missing_dependency_raises_friendly_error(self):
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with self.assertRaises(rag.RAGDependencyError):
                rag.EmbeddingPipeline("x").embed_query("hi")


class TestVectorStore(unittest.TestCase):
    def test_query_clamps_n_results_to_collection_size(self):
        store = rag.VectorStore(Path("tmp"))
        store._ensure = lambda: None  # avoid importing chromadb
        store._collection = MagicMock()
        store._collection.count.return_value = 2
        store._collection.query.return_value = {
            "documents": [["a", "b"]],
            "metadatas": [[{"file_path": "a.py"}, {"file_path": "b.py"}]],
            "distances": [[0.1, 0.2]],
        }
        results = store.query([0.5], n_results=5)
        self.assertEqual(len(results), 2)
        self.assertEqual(store._collection.query.call_args[1]["n_results"], 2)


class TestIndexRepository(unittest.TestCase):
    def test_indexes_with_mocked_dependencies(self):
        tmp, root = _make_repo()
        try:
            cfg = {"retrieval": {}, "indexing": {}}

            def fake_embed(texts):
                return [[0.1, 0.2] for _ in texts]

            with patch("panther.rag.EmbeddingPipeline") as mock_embed, patch(
                "panther.rag.VectorStore"
            ) as mock_store, patch("panther.rag.config.db_path", return_value=Path(tmp.name)):
                mock_embed.return_value.embed_documents.side_effect = fake_embed
                store_instance = mock_store.return_value
                count = rag.index_repository(root, cfg)
                added = store_instance.add_chunks.call_args[0][0]
            self.assertGreater(count, 0)
            self.assertEqual(len(added), count)
            self.assertTrue(all("embedding" in c and "id" in c for c in added))
        finally:
            tmp.cleanup()

    def test_empty_repo_returns_zero(self):
        tmp, root = _make_repo()
        try:
            cfg = {"retrieval": {}, "indexing": {}}
            with patch("panther.rag.EmbeddingPipeline"), patch(
                "panther.rag.VectorStore"
            ), patch("panther.rag.config.db_path", return_value=Path(tmp.name)):
                # Point the scanner at an empty directory.
                with patch("panther.rag.RepoScanner") as mock_scanner:
                    mock_scanner.return_value.scan.return_value = []
                    count = rag.index_repository(root, cfg)
            self.assertEqual(count, 0)
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
