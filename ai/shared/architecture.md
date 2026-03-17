# Architecture

- The repo is script-centered rather than app-centered.
- Playwright-based collection scripts gather raw performance data.
- CSV and checkpoint files are the working data products.
- Regeneration and recheck steps should stay explicit because outputs can be large and long-running to produce.
