# Changelog

All notable changes to Panther CLI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

- Python package metadata and the `panther` console-script entry point.
- Python CLI commands for task runs, analysis, model management, config, sessions, indexing, and skill discovery.
- Python agent/tool loop, permission decision matrix, project-root file tools, and session persistence.
- Optional in-process `llama-cpp-python` local inference adapter.
- Cross-platform Python CI and standard-library unit tests.

### Not yet implemented in the Python runtime

- Semantic-vector RAG, full spec workflow, remote providers/BYOK, native credential storage, and skill-based tool registration.
- Trusted model hashes; the current model manifest entries do not include expected checksums.
