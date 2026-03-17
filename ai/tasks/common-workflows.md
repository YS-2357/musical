# Common Workflows

## Iterate Collection
```bash
python kopis_iterate_pf_playwright.py --start 113846 --end 284000 --delay 0.3 --save-every 200 --resume
```

## Recheck Genre Hint
```bash
python recheck_genre_hint.py --input kopis_iterated.csv --delay 0.5 --log-every 10
```

## Change Rules
- Preserve checkpoint/resume behavior.
- Avoid silent schema changes to generated CSV outputs.
- Document any changes that affect data regeneration steps.
