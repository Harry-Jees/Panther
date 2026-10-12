# Panther — technology choices and rationale

## 1. Language and packaging

Panther is implemented and distributed as a Python CLI package. The installable
command is declared in `pyproject.toml` and points to `panther.cli:main`.
Supported Python starts at 3.10. The application has no Node.js runtime or npm
build step.

## 2. Inference

- Local inference uses `llama-cpp-python`, an optional Python binding to
  llama.cpp. Install it with `pip install "panther-cli[local]"`.
- GGUF weights are downloaded on demand and stored under `~/.panther/models/`;
  they are not bundled in the Python distribution.
- Panther currently supports the local provider only. Remote BYOK providers are
  a future feature and are not advertised as operational.
- Native inference builds and GPU support are platform-dependent; the CLI
  reports a clear error if the optional backend is unavailable.

## 3. CLI and application modules

The command-line interface uses Python's standard-library `argparse`. Core
configuration, sessions, permissions, tool execution, model management, and the
agent loop use the Python standard library. No external server or daemon is
required.

## 4. Storage

- JSON config and session files are human-readable.
- Global state is stored under `~/.panther/`.
- Project sessions, configuration, and the current file inventory are stored
  under `.panther/`.
- GGUF model files are cached globally; model files are excluded from the
  package.

## 5. Testing and CI

The core tests use `unittest` and do not download model weights. CI tests Python
3.10–3.12 across Windows, macOS, and Linux. Local inference smoke tests require
the optional `local` extra and a downloaded model and are separate from the
default test suite.

## 6. PyPI distribution

Build metadata and the `panther` console script live in `pyproject.toml`.
Install locally with `python -m pip install -e .`; install inference support
with `python -m pip install ".[local]"`. Model weights are fetched separately
when a model-backed command first needs them.

## 7. Current limitations

The Python implementation is an early migration, not feature parity with the
former TypeScript prototype. Remote APIs, embeddings/vector RAG, full
requirements/design/tasks orchestration, credential storage, and skill-driven
tool registration remain future work. Current indexing records a project file
inventory; it does not create semantic embeddings.
