# Changelog

All notable changes to Panther CLI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

- **Phase 0: Foundations**
  - TypeScript project scaffold with ESM config (`tsconfig.json`, `vitest.config.ts`, `eslint.config.js`)
  - Config system with Zod validation and project > global > default merge
  - Model manager with resumable GGUF download, progress bar, and SHA-256 verification
  - Local inference provider wrapping `node-llama-cpp` v3 with GPU auto-detection
  - CLI skeleton with `panther run`, `panther model`, `panther config` commands
  - Interactive REPL mode with startup banner
  - Cross-platform CI matrix (Windows, macOS, Linux × Node 20, 22)

- **Phase 1: Core Agent Loop**
  - Tool executor interface and 7 built-in tools (read_file, write_file, apply_diff, list_dir, grep, run_tests, run_command)
  - PermissionGate with 3-mode decision matrix (always_ask, ask_when_risky, never_ask)
  - One-shot auto-denial for non-interactive mode
  - Agent loop with function call parsing and tool dispatch

- **Phase 2: Sessions & Memory**
  - Session manager with Zod schema, JSON file storage, and create/load/save/list
  - `--session <id>` flag support for session resumption
  - Memory system with `memory.md` read/write/append and token budget truncation

- **Phase 3: RAG Context System**
  - Vector store interface with in-memory implementation (sqlite-vec ready)
  - File chunker with function/class-boundary-aware splitting
  - Incremental project indexer with mtime tracking
  - RAG retriever with cosine similarity search and context budget capping
  - `panther index` CLI command

- **Phase 4: Spec-Driven Workflow**
  - SpecWorkflow state machine (requirements → design → tasks → execution → done)
  - Requirements generation in EARS notation
  - Design document generation
  - Task list generation with checkbox items
  - Sequential task execution with checkpoint marking and memory logging

- **Phase 5: Skills System**
  - SKILL.md discovery (project-local and global directories)
  - YAML frontmatter parsing (name, description, triggers, tools, preApproved)
  - Lazy skill loading (full body only on task match)
  - Skill-registered tool registration

- **Phase 6: BYOK Model Providers**
  - Provider registry with `resolveProvider()` factory
  - OpenAI-compatible provider with streaming and tool calling
  - Anthropic Claude provider with streaming and tool calling (SDK stubbed)
  - Google Gemini provider (SDK stubbed)
  - OS-native keychain credential storage via `cross-keychain`

- **Phase 7: Permissions & Polish**
  - Interactive permission config UI
  - First-run setup wizard with model tier selection, permission mode, telemetry opt-in, and BYOK config
  - Model switch confirmation prompt

- **Phase 8: Packaging & Release**
  - GitHub Actions CI configuration
  - Nightly CI workflow with real-model smoke test on self-hosted runner
  - Release workflow with npm publish and GitHub release creation
  - Changelog with conventional commits format
