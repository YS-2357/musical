# Project Context

- Project: musical
- Goal: collect KOPIS musical-category performance data
- Approach: two-stage pipeline
  1. List enumeration via KOPIS internal JSON API (`enumerate_musicals.py`)
  2. Detail page rendering and parsing via Playwright (`kopis_iterate_pf_playwright.py --ids-file`)
- Main inputs: `mt20ids_musical.txt`
- Main outputs (gitignored): `kopis_musical.csv`, `mt20ids_musical_done.txt`, checkpoint, `skipped_musical.jsonl`
- Data collection jobs are resumable; v1 brute-force iterator is archived in `legacy/v1-scraper`.
