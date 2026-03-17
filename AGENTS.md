# Musical Agent Rules

This repo is a KOPIS musical-performance data collection and cleanup workspace.

## Priorities
- Preserve data collection repeatability.
- Avoid breaking resumable scraping and checkpoint-based workflows.
- Keep dataset regeneration steps explicit.
- Before execution, summarize: problem definition -> cause -> solution.

## Commit Rules
- Use a common convention: `type: short english summary`.
- Unless explicitly instructed otherwise, bundle related changes and proceed through `git add`/`git commit`.

## Shared AI Context
- Project context: `ai/shared/project-context.md`
- Architecture: `ai/shared/architecture.md`
- Glossary: `ai/shared/glossary.md`
- Common workflows: `ai/tasks/common-workflows.md`

## Tool Notes
- Tool-specific guidance lives under `ai/tools/`.
- Keep repo-wide shared AI context in `ai/shared/`.
