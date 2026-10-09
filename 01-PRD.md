# Product Requirements Document
## Panther — self-contained, spec-driven local coding agent (working name)

**Status:** Draft v1
**Owner:** Tusker

---

## 1. Problem statement

Every current AI coding agent forces one of two trade-offs:

- **Cloud-dependent** (Kiro, Cursor, Claude Code, Copilot): powerful, but requires an API key, sends code off-device, and stops working without internet or a subscription.
- **Locally flexible but fragmented** (aider, Cline, opencode + Ollama): supports local models, but requires the user to separately install and run a model server (Ollama/LM Studio) before the tool does anything. Setup friction kills adoption for non-power-users.

Nobody currently ships a **single self-contained binary** that does spec-driven agentic coding, runs a real model in-process with zero external dependencies, and remembers project context across sessions without a cloud account.

## 2. Goals

- A developer runs one install command and one setup wizard; from that point on, the tool works fully offline, forever, with no other app required.
- The agent plans before it codes: requirements → design → tasks → implementation, mirroring how a careful engineer actually works — not just "vibe coding" freeform edits.
- The agent remembers what it did in a given project across sessions, without polluting git history for the team unless the user wants that.
- The agent is extensible via user-defined skills, the same pattern power users expect from Claude Code / Kiro / Cline today.
- Permission behavior is configurable, so the tool is trustworthy for both a solo dev who wants speed and a cautious dev who wants to review every change.

## 3. Non-goals (v1)

- No IDE/editor extension (CLI only — see §9).
- No cloud sync, team collaboration, or multi-repo orchestration.
- No custom-trained model — v1 uses existing open-weight GGUF models.
- No support for closed/proprietary model APIs (that's every other tool's feature, not this one's differentiator).

## 4. Target users

- Indie developers and small teams who want agentic coding assistance but are cost-conscious, privacy-conscious, or working offline/air-gapped.
- Developers on teams with strict IP policies that prohibit sending code to third-party APIs.
- Power users of Claude Code/Kiro/aider who want the same workflow without a subscription or internet dependency.

## 5. Key differentiators

| | Kiro | aider + Ollama | Cline | **Panther** |
|---|---|---|---|---|
| Model runs without a separate app | No (cloud) | No (needs Ollama) | No (needs API/Ollama) | **Yes, in-process** |
| Spec-driven (requirements→design→tasks) | Yes | No | No | **Yes** |
| CLI-native | Yes | Yes | No (IDE only) | **Yes** |
| Skills / extensibility | MCP + steering | Limited | MCP | **Skills.md + MCP** |
| Fully offline capable | No | Only if Ollama is local | Only if local model | **Yes, always (default)** |
| Optional cloud model via own key | Built-in (Claude) | Yes, many providers | Yes, many providers | **Yes, opt-in — local stays default** |

## 6. Core features (v1)

1. **Zero-dependency local inference** — model runs in-process via an embedded llama.cpp binding. No daemon, no server, no separate install.
2. **Setup wizard** — first-run flow asks the user to pick a model tier (small/fast vs large/capable); the Qwen coding-model family is the default across both tiers, downloaded as GGUF weights into a local cache. Users can switch models later.
3. **Bring-your-own-key (BYOK) remote models** — optional: the user can add API keys for OpenAI, Gemini, Anthropic, NVIDIA (or any OpenAI-compatible endpoint), and select any of those as the active model for a session instead of the local one. This is explicitly opt-in and never required — Panther's default, out-of-the-box experience stays fully local and self-contained; BYOK is an escape hatch for users who want frontier-model capability on a specific hard task and are willing to trade privacy/offline-ness for it on that occasion.
4. **Session model** — REPL mode for interactive work, one-shot mode for CI/scripts, both reading/writing the same session state.
5. **Spec-driven workflow** — every non-trivial task goes through requirements → design → tasks → implementation, with the user able to review/edit at each stage before the agent proceeds. Trivial tasks (single obvious bug fix) can shortcut this if the user allows it.
6. **memory.md** — per-project, gitignored file recording what the agent did, decisions made, and context worth remembering next session.
7. **Skills system** — user- or community-authored `SKILL.md` files the agent can discover and load, the same pattern as Claude Code/Kiro skills, for teaching the agent project- or domain-specific practices.
8. **Configurable permissions** — dropdown setting (always ask / ask when risky / never ask), settable globally or per-agent/task-type.
9. **General-purpose language support** — no hardcoded stack; works via general code understanding plus optional user-authored skills for specific frameworks.
10. **Branded session start** — launching a session prints a colored "Panther" wordmark/banner to the terminal (see design doc §11), giving the tool a recognizable, polished identity the moment it starts, in the same spirit as Claude Code's startup banner.

## 7. Success metrics

- Time from `pip install "panther-cli[local]"` to first successful agent-completed task (target: under 10 minutes including model download on typical broadband).
- % of tasks completed without the user needing to manually intervene outside the approved permission flow.
- Weekly active CLI sessions per installed user (retention proxy).
- Skill adoption: % of active users who have installed at least one community/custom skill.

## 8. Monetization

Freemium. Core agent, local inference, spec-driven workflow, memory, and skills are free and open. Paid tier (future, not v1) could cover: larger bundled/hosted model access for users without capable local hardware, team-shared skill/memory sync, priority support. This PRD scopes only the free core.

## 9. Future considerations (explicitly out of v1 scope, noted for roadmap)

- VS Code / IDE extension using the same core agent as a library.
- Optional cloud model fallback for users on weak hardware (opt-in, not default).
- Team-shared memory/skills sync (would need to revisit the "gitignored, local-only" memory decision).

## 10. Key risks

- **Model capability ceiling**: spec-driven multi-step reasoning is exactly where smaller local models are weakest. BYOK partially mitigates this — a user stuck on a hard spec can point that one session at a frontier API key — but the default local experience still needs to be credible on its own, since BYOK shouldn't become a crutch the product quietly depends on. Recommend defaulting new users toward the larger Qwen tier with a clear hardware-requirement disclosure, rather than steering them to "small/fast" as the path of least resistance.
- **API key handling**: BYOK means Panther stores and transmits third-party API keys. This must use OS-native secure credential storage (not a plaintext config file) and needs its own clear consent/data-handling disclosure, since it directly touches the privacy-first positioning that's the product's core differentiator.
- **First-run download friction**: multi-GB model downloads before first use. Needs a resumable, progress-visible download with a "you can start with the small model now and upgrade later" escape hatch.
- **Hardware variance**: GPU acceleration (CUDA/Metal/Vulkan) availability varies wildly; CPU-only fallback must still be usable, just slower, and the tool must communicate this honestly rather than silently degrading.
