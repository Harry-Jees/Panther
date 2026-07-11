# Tech Stack Document
## Panther — technology choices and rationale

---

## 1. Language & runtime

**Node.js + TypeScript.** Matches your existing PERN/React skillset directly, has the richest npm-native tooling for CLI distribution, and `node-llama-cpp` (the load-bearing dependency for this whole product) is a first-class Node package with prebuilt binaries — no need to shell out to a separate process.

*Why not C:* the actual inference hot path runs inside llama.cpp regardless of orchestration language — it's the same native binary whether you call it from C, Rust, Python, or Node. Writing the orchestration layer (session management, tool loop, REPL, config) in raw C buys no speed here and costs significant development time (manual memory management, weaker CLI/TUI ecosystem, harder cross-platform builds) for a solo/indie build. *Why not Python:* fine for prototyping, but PyInstaller-style single-binary packaging is heavier and less reliable than npm's distribution model. *Rust* is a legitimate future alternative if single static-binary distribution (zero runtime dependency) becomes a priority — worth revisiting post-v1, not a v1 blocker.

## 2. Inference layer

- **`node-llama-cpp`** — in-process llama.cpp bindings. Handles model loading, GPU backend auto-detection (CUDA/Metal/Vulkan/CPU fallback), grammar-constrained (GBNF) generation for reliable tool-calling, and embeddings, all without a server process.
- **Model format**: GGUF, quantized (start at Q4_K_M as the default balance of size/quality; expose Q5/Q6 for the "large/capable" tier on capable hardware).
- **Default model: Qwen coding-model family**, both tiers — Qwen3-Coder 80B-A3B (or the current-at-build-time equivalent) GGUF quant for "large/capable", a smaller Qwen coder distill for "small/fast". Locked as the default per product decision; exact point-release should still be checked against benchmarks right before build/launch, since Qwen ships frequent updates.
- **Embedding model**: a small dedicated embedding GGUF (e.g., an EmbeddingGemma or a compact BGE variant), loaded separately from the chat model.

**BYOK remote providers** (all optional, behind the `ModelProvider` interface from the design doc):
- `openai` SDK — covers OpenAI directly, and doubles as the client for any OpenAI-compatible endpoint (NVIDIA NIM, and most third-party/self-hosted APIs use this same shape).
- `@anthropic-ai/sdk` — Anthropic.
- `@google/genai` — Gemini.
- **Key storage**: `cross-keychain` (wraps `@napi-rs/keyring` with a CLI fallback) for OS-native keychain storage on Windows/macOS/Linux. No API key ever written to a plaintext config file. `keytar` is deprecated and should not be used.

## 3. CLI framework

- **`commander`** or **`oclif`** for command parsing (`panther run`, `panther model switch`, `panther skill install`, etc.) — oclif if you want strong plugin architecture parity with the skills system; commander if you want something lighter to start.
- **`ink`** (React for CLIs) for the REPL's interactive UI — genuinely a good fit here since it lets your team apply React component thinking directly to a terminal UI (streaming tokens, approval prompts, task-progress views).

## 4. Storage

- **Plain markdown files** for memory.md, requirements.md, design.md, tasks.md — human-readable, git-diffable, no database needed for these.
- **SQLite** (`better-sqlite3`, synchronous and fast for a single-user local tool) + **`sqlite-vec`** extension for the RAG vector index — single file, no server, ships as a native addon.
- **JSON config files** for session state and permission settings.

## 5. Code parsing

- **`tree-sitter`** with WASM or native grammars (native preferred in a Node CLI context, since there's no browser sandbox constraint here) for function/class-boundary-aware chunking across languages.

## 6. Packaging & distribution

- Published as an npm package (`npm install -g panther-cli` or `npx panther-cli`).
- Native addons (`node-llama-cpp`, `better-sqlite3`) ship prebuilt per-platform binaries via their own postinstall mechanisms — verify this at implementation time and pin versions carefully, since native addon breakage across Node versions is the most common failure mode for this kind of tool.
- Model files are **not** bundled in the npm package (would make it enormous) — downloaded on first run per the design doc, cached in `~/.panther/models/`.

## 7. Testing

- **Vitest** for unit tests (fast, TS-native, good fit alongside Node/Ink).
- Integration tests for the agent loop run against a **mocked inference layer** (a fake model that returns scripted tool calls) so the test suite doesn't require downloading multi-GB weights in CI — real-model tests run separately as a smaller, manually-triggered suite.
- Golden-file tests for the spec workflow (given fixed requirements input, does design.md generation stay structurally consistent).

## 8. CI/CD

- **GitHub Actions**: lint + typecheck + unit tests on every PR across Windows/macOS/Linux runners (native addon compatibility is exactly the kind of thing that silently breaks on one OS).
- Separate nightly job running the mocked-model integration suite plus a smaller real-model smoke test on a self-hosted runner with a GPU, since GitHub-hosted runners won't have the hardware for a real inference pass.

## 9. Telemetry (opt-in only)

- If included at all in v1, strictly opt-in, anonymized, and scoped to product metrics only (task completion counts, not code content). This needs its own explicit consent screen in the setup wizard — never default-on, given the privacy-first positioning is a core differentiator.

## 10. Open questions to resolve during implementation

- Exact default model choice per tier (revisit against benchmarks at build time, not now).
- Whether `sqlite-vec` or a lighter pure-JS vector search is sufficient at typical repo sizes — benchmark against a few real large repos before committing.
- Sandbox depth for `run_command` on each OS — start with working-directory confinement, evaluate `bubblewrap`/similar for a v1.1 hardening pass.
