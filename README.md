# Panther CLI

Panther is a local-first coding agent distributed as a Python package. Its CLI,
agent loop, tools, configuration, and session storage are Python-only; it does
not require Node.js or a separately installed local model server.

> **Status:** Pre-alpha. The Python runtime supports local GGUF inference,
> NVIDIA NIM chat completions, and core file/command tools. Vector embeddings
> and the full spec workflow are not yet available.

## Install

Use Python 3.10 or newer. A virtual environment is not required; install
Panther into the Python installation you intend to use:

```bash
python -m pip install panther-cli
```

For a checkout that has not been published to PyPI, install the local package
in editable mode instead:

```bash
python -m pip install -e .
```

For NVIDIA chat, create a `.env` file in the project directory containing:

```dotenv
NVIDIA_API_KEY=your_nvidia_api_key
```

Then select NVIDIA and start an interactive chat:

```bash
python -m panther config --provider nvidia
python -m panther
```

To use a local GGUF model instead, install the optional in-process llama.cpp
backend:

```bash
python -m pip install "panther-cli[local]"
```

`llama-cpp-python` may require platform-specific build or GPU instructions.
Without the `[local]` extra, you can still use NVIDIA NIM for model-backed
tasks, or use the CLI's configuration, sessions, and repository-analysis
commands.

For development:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

## Usage

```bash
# Start an interactive session (local model)
python -m panther

# Run a task with the local model
python -m panther run "Explain the main modules"

# Use NVIDIA NIM with NVIDIA_API_KEY set in the project-root .env file
python -m panther config --provider nvidia
python -m panther

# Select a different NVIDIA model or switch providers
python -m panther config --model nvidia/nemotron-3.5-lightning-30b-a3b
python -m panther config --provider local
python -m panther auth status nvidia

# List models advertised by your NVIDIA API key
python -m panther nvidia models

# Machine-readable output
python -m panther run --json "Summarize the project"

# Manage local model downloads
python -m panther model list
python -m panther model switch small

# Inspect project state
python -m panther analyze .
python -m panther config
python -m panther session list
python -m panther skills
```

Panther stores global settings and downloaded models under `~/.panther`. Project
sessions, configuration, and the local file inventory are stored under the
project's `.panther/` directory. `python -m panther auth add nvidia` prompts for the API
key without echoing it and saves it in the operating system's credential store;
the key is never written to Panther config files. NVIDIA becomes active after
adding the key. Alternatively, set `NVIDIA_API_KEY=your-key` in a `.env` file
in the project root; an already-set `NVIDIA_API_KEY` environment variable takes
precedence. Panther reads only this credential from `.env` and does not export
it to shell commands. `.env` is git-ignored. Get an API key from NVIDIA's
developer portal. NVIDIA requests require internet access and depend on
NVIDIA's model and API availability.

## License

ISC
