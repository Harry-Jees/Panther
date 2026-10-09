# Repository agent guidance

Use this repository to build and maintain Panther as a Python-only local-first
coding agent and CLI, suitable for distribution through PyPI.

## Preferred role
Prefer a repository-analysis mindset when answering questions about this project:
- inspect the current workspace or supplied folder first
- answer using evidence from files, tests, and documentation
- explain architecture, entry points, and implementation paths clearly
- help prepare the project for packaging and future distribution

## Packaging
The application runtime and package entry points are Python. Keep distribution
focused on `pyproject.toml`, pip-installable extras, and cross-platform Python
validation. Do not introduce a Node.js runtime or npm build step.
