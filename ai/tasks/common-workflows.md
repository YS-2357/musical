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

## 3) (필요 시) SKIP 항목 retry + merge

본 추출이 끝난 뒤 `skipped_musical.jsonl` 의 `not_a_record` / `genre_mismatch` 항목이 있으면 한 번 더 돌리고 mt20Id 기준으로 본 CSV 와 병합:

```bash
# retry 입력 만들기
.venv/bin/python -c "
import json
ids = set()
for line in open('skipped_musical.jsonl'):
    r = json.loads(line)
    if r['reason'] in ('not_a_record', 'genre_mismatch'):
        ids.add(r['mt20Id'])
open('retry_ids.txt','w').write('\n'.join(sorted(ids))+'\n')
print(len(ids))
"

# retry 스크랩
.venv/bin/python kopis_iterate_pf_playwright.py \
  --ids-file retry_ids.txt \
  --out-csv kopis_musical_retry.csv \
  --out-ids mt20ids_musical_retry_done.txt \
  --checkpoint kopis_iterate_musical_retry.checkpoint \
  --skipped-log skipped_musical_retry.jsonl \
  --delay 0.5 --save-every 50 --log-every 5

# 병합 (retry 본 우선)
.venv/bin/python -c "
import csv
from kopis_iterate_pf_playwright import FIELDNAMES
rows = {}
for path in ('kopis_musical.csv', 'kopis_musical_retry.csv'):
    with open(path, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            rows[r['mt20Id']] = r
with open('kopis_musical_merged.csv','w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
    w.writeheader()
    for k in sorted(rows):
        w.writerow({c: rows[k].get(c,'') for c in FIELDNAMES})
print('merged rows:', len(rows))
"
mv kopis_musical_merged.csv kopis_musical.csv
```

## 4) Verify / diff
```bash
.venv/bin/python verify_rescrape.py \
  --old <old_csv> --new kopis_musical.csv \
  --out verify_report.md
```

## Change Rules
- Preserve checkpoint/resume behavior (range mode uses int, ids-file mode uses mt20Id string).
- Avoid silent schema changes to generated CSV outputs.
- Document any changes that affect data regeneration steps.
