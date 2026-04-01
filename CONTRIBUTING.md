# Contributing to keeper.raw

Thanks for wanting to help! This guide explains how to contribute to keeper.raw.

## Quick Start

1. Fork the repo and clone your fork
2. Create a branch: `git checkout -b feat/my-feature`
3. Make your changes
4. Run the checks: `cargo clippy --workspace -- -D warnings && cargo test --workspace`
5. Commit using [Conventional Commits](#commit-messages)
6. Push and open a Pull Request

## Where to Start

Look for issues labeled:

- **[`good first issue`](https://github.com/AtharvVatsal/keeper.raw/issues?q=label%3A%22good+first+issue%22)** — small, well-scoped tasks with clear instructions
- **[`help wanted`](https://github.com/AtharvVatsal/keeper.raw/issues?q=label%3A%22help+wanted%22)** — larger tasks where community help is welcome
- **[`ui`](https://github.com/AtharvVatsal/keeper.raw/issues?q=label%3Aui)** — pure TypeScript/React tasks (no Rust needed)
- **[`rust`](https://github.com/AtharvVatsal/keeper.raw/issues?q=label%3Arust)** — pure Rust tasks (no frontend needed)

You don't need to understand the ML pipeline to contribute. Many tasks are isolated to a single crate or the frontend.

## Development Setup

See [docs/development.md](docs/development.md) for full setup instructions.

**Short version:**
```bash
git clone https://github.com/AtharvVatsal/keeper.raw.git
cd keeper.raw
npm install
cargo tauri dev
```

You'll also need the ONNX model files — see [models/README.md](models/README.md).

## Project Structure
crates/
├── keeper-core/      # Shared types, config — start here to understand the data model
├── keeper-ingest/    # File I/O — good for Rust beginners
├── keeper-stacker/   # Burst grouping — algorithmic, well-tested
├── keeper-vision/    # ML pipeline — requires ONNX knowledge
└── keeper-xmp/       # XMP writing — small, self-contained
src/
└── App.tsx           # Entire frontend — React/TypeScript

## Coding Standards

### Rust

- Run `cargo clippy --workspace -- -D warnings` — zero warnings allowed
- Run `cargo fmt --all` before committing
- Add tests for new functionality — run `cargo test --workspace` to verify
- Use `anyhow` for error handling in binaries, `thiserror` for library errors
- Add doc comments (`///`) to all public functions and types

### TypeScript

- Run `npm run build` to verify the frontend compiles
- Keep all UI in `App.tsx` for now (this will be refactored in a future version)
- Use inline styles (no CSS framework yet)
- No external state management — `useState` and `useMemo` only

### Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/). The format is:

type(scope): description

**Types:**

| Type | When to use |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `test` | Adding or updating tests |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `chore` | Build process, CI, dependencies |

**Examples:**
feat(vision): add noise floor estimation to focus scorer
fix(xmp): escape special characters in filenames
docs: update installation guide for Linux
test(stacker): add edge case for single-image scenes
chore(ci): add macOS ARM runner to CI matrix

**Scope** is the crate or area: `core`, `ingest`, `stacker`, `vision`, `xmp`, `ui`, `ci`, `docs`.

## Pull Request Process

1. **One PR = one thing.** Don't mix a bug fix with a feature. Keep PRs focused.
2. **Update tests.** If you change behavior, update or add tests.
3. **Fill out the PR template.** It's there to help reviewers understand your change.
4. **CI must pass.** Your PR won't be reviewed until all checks are green.
5. **Be patient.** This is a solo-maintained project. Reviews may take a few days.

## Reporting Bugs

Use the [Bug Report template](https://github.com/AtharvVatsal/keeper.raw/issues/new?template=bug_report.md). Include:

- What you expected to happen
- What actually happened
- Steps to reproduce
- Your OS, keeper.raw version, and camera model/RAW format

## Suggesting Features

Use the [Feature Request template](https://github.com/AtharvVatsal/keeper.raw/issues/new?template=feature_request.md). Check the [Roadmap](README.md#roadmap) first — your idea might already be planned.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold it.