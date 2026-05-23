# Project Context

- Project: musical
- Goal: collect KOPIS musical-category performance data
- Approach: two-stage pipeline
  1. List enumeration via KOPIS internal JSON API (`enumerate_musicals.py`)
  2. Detail page rendering and parsing via Playwright (`kopis_iterate_pf_playwright.py --ids-file`)
- Inputs (committed): `mt20ids_musical.txt`
- Outputs (committed): `kopis_musical.csv` (21,764 rows as of 2026-05-23)
- Runtime artifacts (gitignored, regenerable): checkpoint, run log, PID, skipped jsonl
- Resumable; v1 brute-force iterator is archived in `legacy/v1-scraper`.
