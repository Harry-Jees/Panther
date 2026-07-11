# Requirements Document

## Introduction

Panther is a self-contained, spec-driven local coding agent. A developer runs one install command and gets a fully offline AI coding assistant that plans before it codes, remembers project context across sessions, and never requires a separate model server or cloud account. Panther runs AI model inference in-process via `node-llama-cpp` bindings, ships as a single npm package, and supports an optional bring-your-own-key (BYOK) mode for remote providers as an opt-in escape hatch.

## Glossary

- **System**: The Panther CLI application process.
- **Agent**: The AI reasoning and tool-calling loop inside the System.
- **ModelProvider**: The abstraction layer that exposes a `generate()` interface, backed by either the local GGUF model or a remote API.
- **LocalProvider**: The `ModelProvider` implementation that uses `node-llama-cpp` in-process bindings.
- **RemoteProvider**: A `ModelProvider` implementation backed by an external API (OpenAI-compatible, Anthropic, or Gemini).
- **SessionContext**: The runtime container holding conversation history, active spec, loaded tools, and the resolved `ModelProvider`.
- **ToolExecutor**: A discrete, independently-testable module that implements a single agent tool (e.g., `read_file`, `run_command`).
- **PermissionGate**: The component that decides whether a tool execution requires user confirmation based on the active permission mode.
- **SpecWorkflow**: The state machine that guides a non-trivial task through requirements → design → tasks → execution stages.
- **Skill**: A `SKILL.md` file plus optional supporting scripts that teach the Agent domain-specific practices.
- **GGUF**: A binary model file format consumed by `node-llama-cpp`.
- **REPL**: Read-Eval-Print Loop; the interactive session mode.
- **BYOK**: Bring-your-own-key; the opt-in mode for using a remote model API with a user-supplied credential.
- **EARS**: Easy Approach to Requirements Syntax; the structured natural-language requirements notation used throughout this document.

---

## Requirements

### Requirement 1: Zero-dependency local inference

**User Story:** As a developer, I want the AI model to run directly inside the Panther process, so that I never need to install or start a separate model server or daemon.

#### Acceptance Criteria

1. THE System SHALL run AI model inference in-process using `node-llama-cpp` bindings with no external server, daemon, or separate application required.
2. THE LocalProvider SHALL auto-detect and use available GPU acceleration (CUDA, Metal, Vulkan) and fall back to CPU-only execution when no GPU is available.
3. WHEN the LocalProvider initialises, THE System SHALL log which GPU backend is active (Metal / CUDA / Vulkan / CPU) to the session output.
4. IF no compatible GPU backend is found, THEN THE LocalProvider SHALL continue using the CPU backend and notify the user that inference will be slower.

---

### Requirement 2: Setup wizard

**User Story:** As a first-time user, I want a guided setup flow when I first run Panther, so that I can configure my model tier, permission mode, and telemetry preference without editing config files manually.

#### Acceptance Criteria

1. WHEN the System starts and no configuration file exists, THE System SHALL present an interactive setup wizard before any other action.
2. WHEN the setup wizard runs, THE System SHALL prompt the user to select a model tier (small/fast or large/capable) and display the estimated disk size and minimum RAM requirement for each tier.
3. WHEN the setup wizard runs, THE System SHALL prompt the user to select a default permission mode from: `always_ask`, `ask_when_risky`, or `never_ask`.
4. WHEN the setup wizard runs, THE System SHALL prompt the user to explicitly opt in or opt out of telemetry before enabling it.
5. WHEN the setup wizard completes, THE System SHALL write the resulting selections to `~/.panther/config.json`.
6. WHEN the setup wizard completes, THE System SHALL initiate a model download for the selected tier with a visible progress bar.
7. IF the user selects the large/capable model tier, THEN THE System SHALL display a hardware-requirement disclosure before confirming the download.

---

### Requirement 3: Model download and cache

**User Story:** As a user, I want Panther to download and cache the selected GGUF model locally, so that all subsequent sessions work fully offline without re-downloading.

#### Acceptance Criteria

1. THE System SHALL download the selected Qwen3-Coder GGUF model from Hugging Face into `~/.panther/models/`.
2. WHEN a model download is in progress, THE System SHALL display a visible progress bar showing bytes downloaded, total size, and estimated time remaining.
3. WHEN a model download is interrupted, THE System SHALL support resuming the download from the point of interruption on the next attempt.
4. WHEN a model download completes, THE System SHALL verify the downloaded file against the expected hash from the model manifest and reject corrupted files.
5. WHEN a model download completes, THE System SHALL write a `manifest.json` to `~/.panther/models/` recording the model name, tier, context window size, and file hash.
6. THE System SHALL expose a `panther model switch <tier>` command that allows the user to change the active model tier after initial setup.
7. WHEN `panther model switch <tier>` is executed and a previous model file exists, THE System SHALL prompt the user for confirmation before deleting the old model file.

---

### Requirement 4: One-shot mode

**User Story:** As a developer using CI pipelines, I want to run Panther with a single task argument and receive a deterministic result, so that I can integrate Panther into automated workflows.

#### Acceptance Criteria

1. THE System SHALL support `panther run "<task>"` for non-interactive single-task execution.
2. WHEN `panther run` completes successfully, THE System SHALL exit with code `0`.
3. WHEN `panther run` fails due to an unrecoverable error, THE System SHALL exit with a non-zero exit code.
4. WHEN `panther run` is invoked with `--json`, THE System SHALL emit all output as newline-delimited JSON to stdout.
5. WHILE operating in one-shot mode, THE PermissionGate SHALL automatically deny any tool execution that would require an interactive approval prompt and log the denial with a descriptive message.
6. THE System SHALL support a `--session <id>` flag on `panther run` to resume an existing session state.

---

### Requirement 5: REPL mode

**User Story:** As a developer doing interactive work, I want a persistent terminal session with streaming output and inline approval prompts, so that I can guide the Agent through multi-turn tasks without losing context.

#### Acceptance Criteria

1. THE System SHALL support `panther` (invoked with no arguments) to enter an interactive REPL session.
2. WHILE in REPL mode, THE System SHALL display a React/Ink terminal UI with a text input component, a streaming token output area, and inline approval prompts.
3. WHEN the Agent emits tokens, THE System SHALL display them as a stream in real time without waiting for the full response.
4. WHEN the Agent requests execution of a risky tool and the permission mode requires a prompt, THE System SHALL display an inline `y/N` approval prompt before proceeding.
5. WHEN a REPL session ends, THE System SHALL persist final session state to `.panther/sessions/<id>.json`.
6. THE System SHALL support `--session <id>` on the `panther` command to resume a previous REPL session.

---

### Requirement 6: Spec-driven workflow

**User Story:** As a developer working on a non-trivial feature, I want the Agent to guide me through requirements, design, and task planning before writing any code, so that I can review and correct the plan at each stage.

#### Acceptance Criteria

1. WHEN a user submits a non-trivial task, THE SpecWorkflow SHALL guide the Agent through stages in order: requirements → design → tasks → execution.
2. WHEN each stage completes, THE SpecWorkflow SHALL write the stage output as a markdown file to `.panther/specs/<spec-id>/` (filenames: `requirements.md`, `design.md`, `tasks.md`) before requesting user approval.
3. WHEN a stage is complete and the output has been written, THE SpecWorkflow SHALL require explicit user approval before advancing to the next stage.
4. WHILE in the requirements stage, THE System SHALL produce requirements in EARS notation.
5. WHEN the execution stage runs, THE System SHALL execute each checkbox task from `tasks.md` sequentially through the Agent loop and mark the checkbox complete in the file after each task finishes.
6. IF the user edits a stage document before approving it, THEN THE SpecWorkflow SHALL use the edited version as the input to the next stage.

---

### Requirement 7: Memory system

**User Story:** As a developer, I want the Agent to remember what it did in my project across sessions, so that I do not have to re-explain project context every time.

#### Acceptance Criteria

1. THE System SHALL maintain a `.panther/memory.md` file per project that is gitignored by default.
2. WHEN a session starts, THE System SHALL read `.panther/memory.md` into the agent context before the first turn.
3. WHEN a session ends, THE System SHALL append a dated session log entry to `.panther/memory.md` summarising files touched, decisions made, and follow-ups noted.
4. WHEN `.panther/memory.md` exceeds a configurable token budget, THE System SHALL truncate it to the most recent entries that fit within the budget before injecting it into context.
5. WHEN `panther run` completes a task in one-shot mode, THE System SHALL also append a session log entry to `.panther/memory.md`.

---

### Requirement 8: Skills system

**User Story:** As a developer, I want to define project-specific or global coding skills that the Agent automatically applies to matching tasks, so that I can encode conventions and best practices once and reuse them across sessions.

#### Acceptance Criteria

1. THE System SHALL discover `SKILL.md` files in `.panther/skills/` (project-local) and `~/.panther/skills/` (global) at session start.
2. WHEN skills are discovered, THE System SHALL load only the YAML frontmatter (`name`, `description`, `triggers`) of each skill without reading the full body.
3. WHEN a task is submitted, THE System SHALL load the full body of a skill into context only when the skill's `description` or `triggers` match the current task.
4. WHERE a skill declares a `tools` array in its frontmatter, THE System SHALL register each listed tool as an available `ToolExecutor` for the session on skill load.
5. THE System SHALL expose a `panther skills list` command that displays discovered skills with their names and descriptions.

---

### Requirement 9: Configurable permission gates

**User Story:** As a developer, I want to configure how much autonomy the Agent has when running tools, so that I can balance speed against safety based on my current context.

#### Acceptance Criteria

1. THE PermissionGate SHALL support three modes: `always_ask`, `ask_when_risky`, and `never_ask`.
2. THE System SHALL read the permission mode from `~/.panther/config.json` as the global default.
3. WHERE a project-level `.panther/config.json` exists, THE System SHALL use its `permissionMode` value in preference to the global default.
4. WHEN the permission mode is `always_ask`, THE PermissionGate SHALL require user confirmation before executing any tool regardless of risk level.
5. WHEN the permission mode is `ask_when_risky`, THE PermissionGate SHALL require user confirmation only before executing tools with `riskLevel: "risky"`.
6. WHEN the permission mode is `never_ask`, THE PermissionGate SHALL execute all tools without prompting.
7. THE System SHALL expose a `panther config permissions` command that allows the user to update the permission mode interactively.

---

### Requirement 10: Tool executors

**User Story:** As a developer, I want the Agent to read, write, and execute code in my project through a well-defined, auditable set of tools, so that I know exactly what actions the Agent can take.

#### Acceptance Criteria

1. THE System SHALL implement the following tools: `read_file`, `write_file`, `apply_diff`, `list_dir`, `grep`, `run_command`, `run_tests`.
2. THE System SHALL classify `write_file`, `apply_diff`, and `run_command` as `riskLevel: "risky"`.
3. THE System SHALL classify `read_file`, `grep`, `list_dir`, and `run_tests` as `riskLevel: "safe"`.
4. WHEN `run_command` executes a shell command, THE System SHALL restrict execution to the project root directory and reject any path traversal attempts.
5. WHEN `apply_diff` is executed, THE System SHALL apply the provided diff to the target file and leave the file in a valid patched state.
6. IF a tool execution encounters an unrecoverable error, THEN THE ToolExecutor SHALL return a structured error result to the Agent loop rather than throwing an unhandled exception.

---

### Requirement 11: RAG context system

**User Story:** As a developer, I want the Agent to retrieve relevant code from my project as context for each task, so that its edits are informed by the broader codebase and not just the file currently in view.

#### Acceptance Criteria

1. THE System SHALL index the project codebase using tree-sitter for function- and class-boundary-aware chunking.
2. THE System SHALL compute text embeddings in-process using a dedicated embedding GGUF model loaded separately from the chat model.
3. THE System SHALL store chunk embeddings in a SQLite database at `.panther/index.db` using the `sqlite-vec` extension.
4. WHEN the project index is built or updated, THE System SHALL re-index only files whose modification time has changed since the last index pass.
5. WHEN the Agent begins a new iteration of the agent loop, THE System SHALL embed the current task string, query the vector store for the top-k most relevant chunks, and inject them into the system prompt as a `<context>` block.
6. THE System SHALL cap the total tokens injected from retrieved context to a configurable fraction of the model's context window.
7. THE System SHALL expose a `panther index` command that triggers an incremental re-index of the project.

---

### Requirement 12: BYOK remote model support

**User Story:** As a developer who occasionally needs frontier-model capability, I want to optionally configure an API key for a remote provider and use it for a specific session, so that I can trade local privacy for higher model quality on hard tasks without making cloud access a permanent default.

#### Acceptance Criteria

1. THE System SHALL support optional BYOK remote model providers: OpenAI-compatible endpoints, Anthropic, and Gemini.
2. THE LocalProvider SHALL remain the default provider when no BYOK configuration is present.
3. WHEN a user adds a BYOK credential, THE System SHALL store the API key in the OS-native keychain and never write it to any plaintext config file.
4. THE System SHALL expose `panther auth add <provider>`, `panther auth remove <provider>`, and `panther auth list` commands for managing credentials.
5. WHEN `panther auth list` is executed, THE System SHALL display which providers have a stored credential without revealing the key values.
6. WHEN a session is started with a remote provider configured, THE System SHALL surface token usage back to the session context so the CLI can display it to the user.
7. IF a requested remote provider has no stored credential, THEN THE System SHALL emit a descriptive error message instructing the user to run `panther auth add <provider>` and exit with a non-zero code.
8. THE System SHALL support selecting the active provider per session via `panther run --model <provider>`.

---

### Requirement 13: Branded session start

**User Story:** As a user, I want to see a recognisable Panther banner when I start a session, so that the tool has a clear, polished identity.

#### Acceptance Criteria

1. THE System SHALL display a "Panther" ASCII art wordmark rendered in amber/yellow at the start of every REPL session.
2. WHEN `panther run` is invoked in one-shot mode, THE System SHALL display the banner before the first line of agent output.

---

### Requirement 14: Packaging and distribution

**User Story:** As a developer, I want to install Panther with a single npm command and have it work immediately on my OS, so that I face no manual build or native-addon compilation steps.

#### Acceptance Criteria

1. THE System SHALL be publishable as an npm package installable via `npm install -g panther-cli`.
2. THE System SHALL ship prebuilt native addon binaries for Windows, macOS, and Linux targeting Node.js 18 and Node.js 20.
3. WHEN the npm package is installed and the native addons fail to load, THE System SHALL display a clear, actionable error message rather than a raw Node.js stack trace.
4. THE System SHALL declare `engines: { "node": ">=18" }` in `package.json` to prevent installation on unsupported Node versions.
5. THE System SHALL pin all native addon dependency versions exactly in `package.json`.
