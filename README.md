# Panther CLI

A self-contained, spec-driven local coding agent that runs entirely offline with no external dependencies.

> **Status:** Pre-alpha (Phase 0: Foundations)

## Install

```bash
npm install -g panther-cli
```

## Usage

```bash
# First run — setup wizard
panther

# Run a one-shot task
panther run "Add error handling to the auth middleware"

# Switch model tier
panther model switch large
```

## Development

```bash
npm install
npm run build
npm test
```

## License

ISC
