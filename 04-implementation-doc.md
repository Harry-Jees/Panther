# Panther — Python implementation status

## 1. Package layout

```text
panther/
├── cli.py             # argparse commands, run mode, and interactive loop
├── config.py          # validated global/project JSON configuration
├── session.py         # JSON session persistence
├── permissions.py     # safe/risky tool approval policy
├── tools.py           # project file and command tools
├── agent.py           # provider/tool dispatch loop
├── provider.py        # optional in-process llama-cpp-python adapter
└── model_manager.py   # resumable GGUF download and local cache
tests/
└── test_core.py       # stdlib unittest coverage; no model download required
```

The installable `panther` command is registered by `pyproject.toml`. Runtime
commands require Python, not Node.js.

## 2. Implemented runtime

- `panther run "<task>"` runs a one-shot task; `--json` produces newline-delimited
  start/completion/error records.
- `panther` starts an interactive loop, and project sessions can be listed,
  shown, resumed, or deleted.
- Config precedence is project `.panther/config.json` over global
  `~/.panther/config.json` over built-in defaults.
- Safe tools include reading/listing/searching files. File writes and shell
  commands are risky and governed by the permission mode. One-shot mode denies
  actions that would require an interactive approval.
- File-writing tools resolve paths under the project root and reject traversal.
- Model downloads are cached globally and resume partial downloads. The
  built-in manifest currently has no published expected hashes, so checksum
  comparison is unavailable until trusted hashes are added.
- Local inference is optional and supplied by `llama-cpp-python`; Panther does
  not invoke an external model server.

## 3. Commands

```text
panther [--version]
panther run [task] [--json] [--session ID]
panther analyze [path]
panther config [--permissions]
panther model list
panther model switch small|large [--yes]
panther session [list|show|delete] [ID]
panther index [--path PATH]
panther skills
```

## 4. Test and development commands

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
```

To enable local model inference:

```powershell
python -m pip install -e ".[local]"
```

The standard test suite uses fake providers and temporary files; it does not
download model weights or require the local inference extra.

## 5. Migration roadmap

1. Complete the Python-only CLI/runtime transition and keep all packaging,
   installation, and CI entry points Python-based.
2. Expand test coverage for model download resume/hash verification and command
   approval behavior.
3. Implement persistent semantic indexing and retrieval.
4. Add the reviewable requirements → design → tasks workflow.
5. Add skill loading/tool registration and optional remote providers only after
   defining their dependency and credential-storage requirements.

The former TypeScript implementation is no longer the application runtime.
