# Panther CLI

Panther is a local-first coding agent distributed as a Python package. Its CLI,
agent loop, tools, configuration, and session storage are Python-only; it does
not require Node.js or a separately installed local model server.

> **Status:** Pre-alpha. The Python runtime supports local GGUF inference,
> named NVIDIA NIM, Google Gemini, and OpenAI-compatible API bots, plus core
> file/command tools. Vector embeddings and the full spec workflow are not yet
> available.

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

## Configure a named API bot

Run the setup wizard in an interactive terminal:

```bash
python -m panther setup
```

The same wizard is available as `python -m panther bot add [NAME]`.

Give the bot a name, choose NVIDIA NIM, Google Gemini, or an OpenAI-compatible
API, enter its model (and API base URL for a compatible endpoint), then enter
the API key through the hidden prompt. Panther stores named-bot keys in the
operating system credential store, not in project config files. The wizard
selects the bot and runs a sample request to verify the connection.

```bash
python -m panther
python -m panther bot list
python -m panther bot test
python -m panther bot use my-gemini
```

Bot profiles are project-specific. `bot add` creates/selects a profile;
`bot use NAME` switches profiles, and `bot remove NAME` removes one after
confirmation. Generic endpoints should use the API's base URL (for example,
`https://api.example.com/v1`), not its `/chat/completions` route.

For unattended use or legacy single-provider config, API keys can also be
provided through environment variables or the project `.env` file:
`NVIDIA_API_KEY`, `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), and
`OPENAI_COMPATIBLE_API_KEY`. Environment variables take precedence over
keyring credentials. `.env` is git-ignored.

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
# Open the native desktop workspace
panther-desktop
# Or launch it from the CLI
python -m panther gui

# Start an interactive session (local model)
python -m panther

# Run a task with the local model
python -m panther run "Explain the main modules"

# Force the dedicated coding model for implementation work
python -m panther run --code "Implement the missing parser"

# Ask the vision model about an image inside the project
python -m panther run --image assets/diagram.png "Explain this architecture"
# A visual question that names one project image attaches it automatically
python -m panther run "Describe what assets/diagram.png shows"

# Use a named provider bot
python -m panther bot add my-nvidia
python -m panther bot use my-nvidia
python -m panther

# Select a different NVIDIA model or switch legacy providers
python -m panther config --model nvidia/nemotron-3.5-lightning-30b-a3b
python -m panther config --code-model openai/gpt-oss-20b
python -m panther config --vision-model google/gemma-4-31b-it
python -m panther config --provider local
python -m panther auth status nvidia

# List models advertised by your NVIDIA API key
python -m panther nvidia models

# In an interactive session, use /image and browse_web when needed
# /image assets/diagram.png Explain this diagram

# Machine-readable output
python -m panther run --json "Summarize the project"

# Manage local model downloads
python -m panther model list
python -m panther model switch small

# Inspect project state
python -m panther analyze .
python -m panther index --path .
python -m panther config
python -m panther session list
python -m panther skills
python -m panther skills add C:\path\to\my-skill
python -m panther skills add https://github.com/example/agent-skills.git --subdir skills/research
python -m panther skills add C:\path\to\SKILL.md --global

# Connect Panther to your local MySQL database
python -m pip install "panther-cli[mysql]"
python -m panther mysql setup --host 127.0.0.1 --port 3306 --database my_app --username panther
python -m panther mysql test
python -m panther mysql status
```

The Windows desktop workspace follows a VS Code-style layout: project
explorer, UTF-8 editor, and Panther agent side by side, with a docked terminal
panel below. Choose a folder to set the workspace root for the editor, agent,
and terminal commands. After each task, Panther prints changed files and
colored unified line diffs inline in the agent pane; sensitive files such as
`.env` are listed without exposing their contents. Risky agent tool actions
still require approval from the app. Terminal commands run in the selected
folder and their output appears in the terminal panel. The desktop launcher is
created as a Windows GUI entry point; the existing `panther` CLI remains
available separately.

On a fresh desktop install, starting `panther` in an interactive terminal asks
for an NVIDIA API key if one is not already configured, loads the models
available to that key, and lets you choose a model before asking a short
onboarding survey. The key and profile fields are stored in the operating
system credential store; they are not put in Panther's JSON configuration or
session history. Profile details stay local unless you explicitly agree to
share them with NVIDIA. Existing installations without an onboarding marker
are not forced through this first-run flow.

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

In an interactive terminal, type `/` to open the slash-command completion menu.
Commands can be selected with the keyboard or mouse. Non-interactive terminals
automatically fall back to plain input. The repository index is built in parallel
and reused incrementally; use `/index` or `python -m panther index --path .` to
rebuild it. Panther can inspect Git status and diffs, and its dedicated Git
tools can push or pull the current branch using its configured upstream. Those
tools always require a fresh interactive approval, including when Autopilot is
enabled; one-shot `panther run` cannot approve them. Shell commands continue to
follow the configured `run_command` permission policy.

Skills use the `SKILL.md` format. Add a Markdown skill file, a folder containing
`SKILL.md` and optional reference files, or a Git repository containing one or
more skills. For multi-skill repositories, choose a skill folder with `--subdir`.
Skills are installed per project under `.panther/skills/` by default, or under
`~/.panther/skills/` with `--global`; `python -m panther skills` lists both.
Installed skill instructions are supplied to Panther and its workers when
handling tasks, and those agents can use the public-web `browse_web` tool when
the skill or task calls for research. Interactive sessions also support
`/skills` and `/skill add <source> [--subdir PATH] [--name NAME] [--global]`.
Install skills only from sources you trust: their Markdown is added to the
agent's instructions, though the installer does not execute skill files.

## MySQL access

Install the optional connector, then run `python -m panther mysql setup` (or
`/mysql setup` interactively). Setup prompts for the password without echoing it,
tests the connection, stores the password in the operating system keyring, and
stores only the host, port, database, and username under the ignored project
directory `.panther/mysql.json`. Use `mysql test`, `mysql status`, or
`mysql remove` to manage the connection.

When configured, Panther can inspect the database with bounded, parameterized,
read-only SQL and can make requested data or schema changes. Each modifying
statement requires a fresh interactive approval, even when Autopilot is on or
the tool was previously remembered; one-shot `panther run` cannot approve a
database write. Panther accepts one statement at a time, rejects SQL account/
server/database administration commands, and does not ask for database passwords
in chat. Values bound through SQL parameters are hidden in the approval prompt.
Use a database account with only the privileges Panther needs.

Risky tools ask for approval by default. Choose `a` at an approval prompt to
remember that tool's approval for the current project; Panther stores these
project-specific grants in `.panther/config.json`. Use `/autopilot` to enable or
disable project-level Autopilot mode, which allows all tools without approval.
Use `/autopilot on` or `/autopilot off` to set it explicitly, and `/status` to
check its state. Autopilot is off by default.

Coding tasks are split into three named workers (Mochi, Boba, and Pip). They
implement independent slices in separate temporary project copies; a lead agent
reviews and integrates their proposed diffs into the real project, then performs
the normal validation workflow. Coding workers and the lead must successfully use
repository inspection, editing, and validation tools before they can report
completion; read-only review workers must inspect the project with a read tool.
With NVIDIA active, Panther discovers models
available to that API key and assigns distinct model IDs when the catalog
offers them, favoring Kimi when the account's model catalog includes it. If the
coding planner returns malformed JSON, Panther asks it once to correct the plan
and then falls back to a built-in three-way task split if needed. In interactive
terminals, the coding UI temporarily shows each worker's live file
reads, searches, edits, and tool results, then clears the activity panel before
the final answer. Redirected output keeps a readable activity log. The UI also
shows a short, real post-build chat: each worker reads the team's implementation
notes before replying, and the lead receives those observations for integration.
Temporary worker copies exclude `.env*`,
`.panther/`, and virtual environment directories. Other providers use the
selected model for each worker. NVIDIA requests are subject to the account's
model access, rate limits, and usage terms.

Ask what the agents suggest building (for example, “What do the three agents
suggest I build?”) to start a suggestion-only brainstorm. Each agent proposes
an idea in turn, sees earlier suggestions to avoid duplicates, and makes no
file changes. Requests to analyze or review a local file, folder, or project
also start the three agents in read-only mode; their tool activity and findings
appear under their names, and they cannot edit files or run commands.

## License

ISC
