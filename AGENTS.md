# Musical Agent Rules

This repo collects KOPIS musical-category performance data via:
1. KOPIS internal JSON API (`enumerate_musicals.py`) for mt20Id enumeration
2. Playwright-rendered detail pages (`kopis_iterate_pf_playwright.py --ids-file`) for record extraction

## Priorities
- Preserve resumability and checkpoint-based collection (works for both range and ids-file modes).
- Avoid breaking the deterministic genre detection in `extract_genre_badge` (reads `<span class*="DBDetail_cls">`).
- Keep `mt20ids_musical.txt` as the canonical input list; regenerate via `enumerate_musicals.py` when KOPIS site updates.
- Before execution, summarize: problem definition -> cause -> solution.

## Commit Rules
- Use a common convention: `type: short english summary`.
- Unless explicitly instructed otherwise, bundle related changes and proceed through `git add`/`git commit`.

## Files

### Code
- `enumerate_musicals.py` — JSON API list collector
- `kopis_iterate_pf_playwright.py` — Playwright detail scraper (range / auto-ceiling / ids-file modes)
- `verify_rescrape.py` — old/new CSV diff report

### Data inputs
- `mt20ids_musical.txt` — enumerated musical mt20Ids

### Generated outputs (gitignored)
- `kopis_musical.csv` — main result
- `mt20ids_musical_done.txt` — completed mt20Ids
- `kopis_iterate_musical.checkpoint` — last processed mt20Id
- `skipped_musical.jsonl` — skip/anomaly log
- `run_musical.log`, `run_musical.pid` — background-run artifacts

### Branch
- `legacy/v1-scraper` — frozen archive of v1 brute-force PF iterator and its 142k CSV.

## Shared AI Context
- Project context: `ai/shared/project-context.md`
- Architecture: `ai/shared/architecture.md`
- Glossary: `ai/shared/glossary.md`
- Common workflows: `ai/tasks/common-workflows.md`

## Tool Notes
- Tool-specific guidance lives under `ai/tools/`.
- Keep repo-wide shared AI context in `ai/shared/`.
