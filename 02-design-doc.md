# Design Document
## Panther — system architecture and subsystem design

---

## 1. High-level architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Panther CLI process                     │
│                                                                │
│  ┌──────────────┐   ┌──────────────────┐   ┌──────────────┐ │
│  │  Session      │   │  Spec workflow    │   │  Permission   │ │
│  │  manager      │◄──┤  state machine    │◄──┤  gate         │ │
│  │  (REPL/       │   │  (req→design→     │   │  (config-     │ │
│  │  one-shot)    │   │  tasks→exec)       │   │  driven)      │ │
│  └──────┬───────┘   └────────┬─────────┘   └───────┬──────┘ │
│         │                    │                       │        │
│         ▼                    ▼                       ▼        │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              Agent loop (tool-calling core)              │  │
│  └──────┬──────────────┬───────────────┬─────────────┬────┘  │
│         │              │               │             │        │
│         ▼              ▼               ▼             ▼        │
│  ┌───────────┐  ┌────────────┐  ┌───────────┐  ┌──────────┐ │
│  │ Model      │  │ RAG index   │  │  Skills    │  │ Memory   │ │
│  │ provider   │  │ (embed +    │  │  loader    │  │ (memory  │ │
│  │ (local or  │  │ vector db)  │  │ (SKILL.md) │  │ .md)     │ │
│  │ BYOK)      │  │             │  │            │  │          │ │
│  └───────────┘  └────────────┘  └───────────┘  └──────────┘ │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Tool executors: read_file, write_file, apply_diff,      │  │
│  │  run_command (sandboxed), grep, list_dir, run_tests       │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
   Local filesystem: project repo, .panther/ (memory.md, skills, index.db, config)
```

Everything lives in one OS process. No network calls except the one-time model download and optional telemetry (opt-in). This is the load-bearing constraint the whole design serves.

## 2. Session model

A **session** is a unit of agent work with its own conversation history, active spec (if any), and a live handle to the loaded model.

- **REPL mode**: `panther` with no args opens an interactive session. Model weights load once at session start and stay resident in memory for the session's lifetime — this is where the "keep it warm" trade-off from the tech discussion pays off.
- **One-shot mode**: `panther run "<task>"` or `panther task <id>` loads the model, executes to completion or to the next approval gate, writes results, and exits. Designed for CI: deterministic exit codes, machine-readable `--json` output mode.
- Both modes read/write the same `.panther/session/<id>.json` state file, so a one-shot invocation can resume a REPL-started task and vice versa.

## 3. Spec-driven workflow (state machine)

```
  [idle] → user request
     │
     ▼
  [requirements]  — agent drafts requirements in EARS-like notation,
     │               user reviews/edits, approves to proceed
     ▼
  [design]        — agent proposes architecture/approach given the
     │               requirements + retrieved codebase context,
     │               user reviews/edits, approves to proceed
     ▼
  [tasks]         — agent breaks design into ordered, checkable tasks
     │               user reviews/edits, approves to proceed
     ▼
  [execution]     — agent loop executes tasks one at a time, each
     │               task gated by the permission system
     ▼
  [done]          — memory.md updated, session summary written
```

Each stage's output is a plain markdown file under `.panther/specs/<spec-id>/` (`requirements.md`, `design.md`, `tasks.md`) — human-editable, diffable, and the exact context fed into the next stage's prompt. A trivial task (explicit single bug fix, user says "just fix it") can skip straight to a single-task execution stage if the user's permission config allows shortcutting.

## 4. Agent loop / tool-calling design

The loop is a standard ReAct-style cycle: model reasons → emits a tool call → executor runs it → result appended to context → repeat until task completion or a permission gate blocks it.

Tool calls are **grammar-constrained** at the token-sampling level (GBNF grammar matching a fixed JSON schema per tool), not just prompted for — this is what makes tool-calling reliable with smaller open models that weren't necessarily fine-tuned for strict function-calling formats.

Tool set (v1): `read_file`, `write_file`, `apply_diff`, `list_dir`, `grep`, `run_command`, `run_tests`. Each tool executor is a small, independently testable module — new tools (and skills that register tools) plug into the same interface.

## 5. Memory system

`.panther/memory.md` per project, gitignored by default (per product decision). Structure:

```markdown
# Project memory

## Architecture notes
(agent-maintained summary of the codebase's structure, updated as it learns)

## Session log
### 2026-07-11 — Fixed null pointer in auth middleware
- Files touched: src/auth/middleware.ts
- Decision: chose optional chaining over a guard clause to match existing style
- Follow-up noted: token refresh logic still untested

### 2026-07-10 — Implemented dark mode toggle
...
```

The agent reads this file into context at session start (summarized/truncated if large) so it doesn't relearn the codebase from scratch each time. Writing to it is itself a tool call (`update_memory`), gated by the same permission system as file edits, but typically pre-approved since it never touches source code.

## 6. Skills system

A skill is a `SKILL.md` file (same pattern as Claude Code/Kiro skills) plus an optional folder of supporting scripts/templates, living in `.panther/skills/<name>/` (project-local) or `~/.panther/skills/<name>/` (global, available across all projects).

- Each `SKILL.md` has frontmatter (`name`, `description`, `triggers`) and a body of instructions/best-practices, exactly like the skill-reading pattern used elsewhere in this ecosystem.
- At session start, the agent scans available skills' descriptions (cheap, metadata-only) and only loads a skill's full body into context when its description matches the current task — same lazy-loading principle as any large skill library, necessary because local model context windows are smaller than cloud frontier models'.
- Skills can register additional tools (e.g., a "Flutter" skill could register a `flutter_analyze` tool that shells out to `dart analyze`), making the system extensible without changing core code.

## 7. Permission system

A single config value, settable globally (`~/.panther/config.json`) or per-project (`.panther/config.json`), with three states: `always_ask`, `ask_when_risky`, `never_ask`. "Risky" is a fixed classification per tool (file writes and shell commands are risky; reads, greps, and listing are not) — not model-judged, so it's predictable and can't be argued around by a persuasive-sounding agent response. Exposed in the setup wizard as a dropdown per the product decision, and overridable per-skill (a skill can declare its own tools as pre-approved when installed, matching the "pre-approved trusted tools" pattern from comparable tools).

## 8. RAG / context system

Reuses the design from the earlier browser-based version, adapted to local storage:

- Chunking via tree-sitter grammars (function/class-boundary aware, per language).
- Embeddings computed in-process via the same llama.cpp binding, using a small dedicated embedding model (separate from the chat model, loaded only during indexing to avoid competing for memory with the chat model during a session).
- Vector storage in SQLite with the `sqlite-vec` extension — a single-file local vector index, no separate database service.
- Incremental re-indexing on file save (only changed files re-chunked/re-embedded), not a full repo re-scan each session.

## 9. Model management

- Setup wizard downloads the chosen tier's GGUF file from Hugging Face directly into `~/.panther/models/`, resumable, with a visible progress bar and a fallback to the smaller tier if the download stalls repeatedly.
- Model metadata (context window, quantization, capability tier) stored alongside the file so the agent loop can adapt prompt sizing to the loaded model's real context limit.
- `panther model switch <tier>` lets the user change tiers later without redoing the whole wizard.

## 10. Model provider abstraction (local + BYOK)

Panther's agent loop calls a single `ModelProvider` interface — everything upstream (agent loop, spec workflow, RAG) is unaware of whether generation happens locally or via a remote API. This is what keeps BYOK an optional add-on rather than a fork in the architecture.

```
interface ModelProvider {
  generate(history, tools): AsyncIterable<TokenOrToolCall>;
  embedText?(text): number[];   // only local/some providers support this
  contextWindow: number;
  supportsGrammar: boolean;      // GBNF-constrained tool calling
}
```

- **LocalProvider**: the Python runtime optionally wraps `llama-cpp-python` and defaults to a Qwen coding-model GGUF. Grammar-constrained tool calling is not yet implemented in the Python runtime.
- **RemoteProvider**: one implementation per API shape — OpenAI-compatible (covers OpenAI, NVIDIA NIM, and most third-party endpoints), Anthropic Messages API, Google Gemini API. Tool calling uses each provider's native function-calling format instead of GBNF, normalized back into the same internal `ToolCall` shape the agent loop expects.
- **Provider selection**: set per-session (`panther run --model gpt-4o-mini`) or as a project/global default in config. Falls back to the local provider if no key is configured for the requested remote provider, with a clear error rather than a silent failure.
- **API key storage**: OS-native secure credential storage (`cross-keychain`, which wraps `@napi-rs/keyring` with a CLI fallback), never plaintext in `.panther/config.json`. Config stores only *which* provider is active and a reference to the stored credential, not the key itself. `keytar` is deprecated and should not be used.
- **Cost/rate-limit awareness**: RemoteProvider implementations surface token usage back to the session so the CLI can show the user what a BYOK session is costing them — important since this is real money per call, unlike the local default.

## 11. Security / sandboxing

- `run_command` executes in a restricted shell context: working directory locked to the project root, environment variable allowlist, and (where the OS supports it) a lightweight sandbox (e.g., `bubblewrap` on Linux, App Sandbox-style restriction on macOS where feasible) — v1 can start with working-directory confinement and expand sandboxing depth in a later pass, but this must be documented honestly rather than implied to be stronger than it is.
- No code from the project or from skills executes with network access by default; skills that need network tools must declare it explicitly and get user approval on install.
