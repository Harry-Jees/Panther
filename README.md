# Panther CLI

Panther is a command-line application that turns any local codebase into an
interactive learning environment. It indexes your repository locally, retrieves
relevant code chunks with an embedding-based RAG pipeline, and answers natural
language questions about your code with a teacher-toned LLM - "viva" sessions.

## Features

- **Interactive viva sessions** - ask questions about your own codebase in plain
  English; Panther explains with referenced code snippets and follow-ups.
- **Local RAG pipeline** - scans the repo (skipping `.git`, `node_modules`,
  etc.), chunks code (500-token chunks, 50-token overlap), embeds with
  `sentence-transformers` (`all-MiniLM-L6-v2`), and stores vectors in a
  persistent local Chroma DB.
- **Free LLM tier** - NVIDIA NIM (OpenAI-compatible endpoint, exponential
  backoff on 429/5xx, rolling tokens-per-minute cap).
- **Agent tools** - Panther can inspect your code while answering: `read_file`,
  `list_dir`, `grep` and `run_tests` (safe, read-only, path-constrained to the
  project). Toggle with `/tools` inside a session.
- **Reranking (optional)** - re-scores retrieved chunks with a local
  cross-encoder for better precision. Enable `"rerank": true` under
  `retrieval` in `~/.panther/config.json`; degrades gracefully if the model
  isn't downloaded yet.
- **Animated UI** - loading spinners, typewriter text-fill, live tool-call
  feedback and streamed markdown answers (with syntax highlighting).
- **Graceful degradation** - no API key or missing optional deps? Panther falls
  back to lightweight keyword retrieval instead of failing.
- **Session persistence** - transcripts saved under `~/.panther/sessions/`;
  resume any prior session with `panther resume <session-id>`.
- **Clean CLI aesthetics** - gradient ASCII banner, rich tables, syntax-aware
  code blocks (when Pygments is installed).

## Install

```bash
# Core (includes RAG + LLM deps; sentence-transformers pulls in torch, so the
# install can take a few minutes)
python -m pip install .

# Optional extras: precise token counting (tiktoken) and syntax highlighting
python -m pip install .[extra]
```

Requires Python 3.9+. For a lighter install that keeps keyword retrieval but
defers the RAG deps, install just `rich` and `requests` and Panther will tell
you what to install when you first run `panther index`.

## Quick start

```bash
panther init                 # create ~/.panther and a default config
panther auth add nvidia_nim  # free key from https://build.nvidia.com
cd /path/to/your/project
panther index                # scan + chunk + embed the repo
panther new                  # start a viva session (auto-indexes if needed)
```

API keys can also be supplied via the `NVIDIA_NIM_API_KEY` environment
variable (it takes precedence over the config file).

## Commands

| Command | Description |
| --- | --- |
| `panther init` | Initialize `~/.panther` and write the default config |
| `panther index` | Index the current repository (keyword + vector) |
| `panther new [--no-index]` | Start a new interactive session |
| `panther resume <session-id>` | Resume a saved session |
| `panther list` | List saved sessions |
| `panther delete <session-id>` | Delete a saved session |
| `panther reset --confirm` | Remove all Panther data |
| `panther config [section]` | Show configuration (or one section, e.g. `llm`) |
| `panther auth add/remove/list` | Manage API keys |
| `panther --version` | Show version |

## Chat interface

On an interactive terminal, `panther new` opens a full-screen chat UI
(opencode/Freebuff-style): a scrollable transcript with the **input bar pinned
at the bottom**, `@file` mentions with tab-completion, and a `/` command
palette. When input is piped (scripts, CI), Panther falls back to the classic
line-based REPL automatically. Set `PANTHER_NO_TUI=1` to force the classic
REPL on a terminal.

### `/` commands

Inside a session: `/help`, `/status`, `/model`, `/tools`, `/reindex`, `/clear`, `/exit`, `/quit`.

### `@file` mentions

Type `@` and press Tab to complete a file path from the repository. The
file's content is attached to your question so the model reads the exact
source you referenced — e.g. `what does @panther/api.py do?`.

Answers fill in with a typewriter effect and live markdown + syntax highlighting.
When the agent uses tools, each `read_file` / `grep` / `list_dir` / `run_tests`
call is shown with a spinner and a status marker. Sessions show a compact header
with the active model and per-turn token usage.

## Configuration

`~/.panther/config.json` holds the LLM provider (model, temperature, max
tokens), retrieval settings (chunk size, overlap, top-k, similarity
threshold), token budgeting, indexing rules and UI preferences. See
`panther config` to inspect the active values.

## Development

```bash
python -m unittest discover -s tests   # run the test suite
```

## License

MIT.
