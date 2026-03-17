# AI Files

This folder stores shared AI context for the `musical` repo.

## Layout
- `shared/`: project facts multiple AI tools can reuse
- `tasks/`: repeatable workflows
- `tools/`: tool-specific notes when behavior differs by AI service

## Usage
- Codex reads the repo root `AGENTS.md`.
- Other AI tools can read from this folder directly or through small adapters later.
- Prefer updating shared facts here instead of duplicating them across tool-specific files.
