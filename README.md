# Panther CLI

Panther is a local-first coding agent distributed as a Python package. Its CLI,
agent loop, tools, configuration, and session storage are Python-only; it does
not require Node.js or an external model server.

> **Status:** Pre-alpha. The Python runtime currently supports local GGUF
> inference and core file/command tools. Remote providers, vector embeddings,
> and the full spec workflow are not yet available.

## Install

Install the CLI and the optional in-process llama.cpp backend:

```bash
python -m pip install "panther-cli[local]"
```

`llama-cpp-python` may require platform-specific build or GPU instructions.
Without the `[local]` extra, the CLI, configuration, sessions, and repository
analysis commands still work, but model-backed tasks report that the backend is
missing.

For development:

```bash
python -m pip install -e ".[local]"
python -m unittest discover -s tests -v
```

## Usage

```bash
# Start an interactive session
panther

# Run a task (risky tools are denied if they would need approval)
panther run "Explain the main modules"

# Machine-readable output
panther run --json "Summarize the project"

# Manage local model downloads
panther model list
panther model switch small

# Inspect project state
panther analyze .
panther config
panther session list
panther skills
```

Panther stores global settings and downloaded models under `~/.panther`. Project
sessions, configuration, and the local file inventory are stored under the
project's `.panther/` directory.

## License

ISC
