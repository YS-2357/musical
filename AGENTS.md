# Musical Agent Rules

This repo collects KOPIS musical-category performance data via:
1. KOPIS internal JSON API (`enumerate_musicals.py`) for mt20Id enumeration
2. Playwright-rendered detail pages (`kopis_iterate_pf_playwright.py --ids-file`) for record extraction

## Priorities
- Preserve resumability and checkpoint-based collection (works for both range and ids-file modes).
- Avoid breaking the deterministic genre detection in `extract_genre_badge` (reads `<span class*="DBDetail_cls">`).
- `mt20ids_musical.txt` is the canonical input; regenerate via `enumerate_musicals.py` when KOPIS site updates.
- Before execution, summarize: problem definition -> cause -> solution.

## Commit Rules
- Use a common convention: `type: short english summary`.
- Unless explicitly instructed otherwise, bundle related changes and proceed through `git add`/`git commit`.

## Files

### Code (committed)
- `enumerate_musicals.py` — JSON API list collector
- `kopis_iterate_pf_playwright.py` — Playwright detail scraper (range / auto-ceiling / ids-file modes)
- `verify_rescrape.py` — old/new CSV diff report

### Data (committed)
- `mt20ids_musical.txt` — enumerated musical mt20Ids
- `kopis_musical.csv` — collected musical performance records (21,764 rows)

### Runtime artifacts (gitignored, regenerable)
- `*.checkpoint`, `*.log`, `*.pid`, `skipped*.jsonl`, `run_musical.*`

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
