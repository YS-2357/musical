# Architecture

- The repo is script-centered rather than app-centered.
- Two-stage collection pipeline:
  - **Stage 1** — Internal KOPIS JSON API (`https://kopis.or.kr:9001/api/prs/v1/por/db/prfrdb/perfo-infos?tabno=ggga`) is hit page by page with `requests`. Output: flat `mt20ids_musical.txt`.
  - **Stage 2** — Playwright (headless Chromium) renders each `pblprfrView.do?mt20Id=...` detail page. `parse_label_value_blocks` over `<dl><dt><dd>` extracts fields; `extract_genre_badge` reads the deterministic `<span class*="DBDetail_cls">` for genre.
- Persisted artifacts: `mt20ids_musical.txt` (input list) and `kopis_musical.csv` (final result). Both are tracked in git.
- Runtime artifacts (checkpoint, log, PID, skipped jsonl) are gitignored — they exist only during/after a single run and are safe to delete.
- Regeneration and recheck steps should stay explicit because the detail run is multi-hour and rate-sensitive (`--delay 0.3` default).
