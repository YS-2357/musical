# Common Workflows

## 1) Enumerate musical mt20Ids
```bash
.venv/bin/python enumerate_musicals.py \
  --out mt20ids_musical.txt \
  --page-size 100 --delay 0.3
```

Switches: `--tabno=<code>` to fetch a non-musical genre (default `ggga` = 뮤지컬).

## 2) Detail scrape (background)
```bash
nohup .venv/bin/python -u kopis_iterate_pf_playwright.py \
  --ids-file mt20ids_musical.txt \
  --out-csv kopis_musical.csv \
  --out-ids mt20ids_musical_done.txt \
  --checkpoint kopis_iterate_musical.checkpoint \
  --skipped-log skipped_musical.jsonl \
  --delay 0.3 --save-every 200 --log-every 100 --resume \
  > run_musical.log 2>&1 &
echo $! > run_musical.pid
```

Monitor: `tail -f run_musical.log`, `wc -l kopis_musical.csv mt20ids_musical_done.txt`, `cat kopis_iterate_musical.checkpoint`.

Stop/resume: `kill "$(cat run_musical.pid)"` (next save_every flushes), then re-run the same `nohup` line.

## 3) Verify / diff
```bash
.venv/bin/python verify_rescrape.py \
  --old <old_csv> --new kopis_musical.csv \
  --out verify_report.md
```

## Change Rules
- Preserve checkpoint/resume behavior (range mode uses int, ids-file mode uses mt20Id string).
- Avoid silent schema changes to generated CSV outputs.
- Document any changes that affect data regeneration steps.
