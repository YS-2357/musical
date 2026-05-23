# musical

KOPIS 공연 데이터(뮤지컬 카테고리) 수집/정리 작업 공간.

## 결과
- `kopis_musical.csv` — 뮤지컬 21,764건 (2026-05-23 수집)
- `mt20ids_musical.txt` — 입력 mt20Id 목록 (enumerate 결과)

## 코드
- `enumerate_musicals.py` — KOPIS 내부 JSON API 로 뮤지컬 mt20Id 목록 수집
- `kopis_iterate_pf_playwright.py` — Playwright 기반 detail 스크래퍼 (`--ids-file` 모드)
- `verify_rescrape.py` — 두 CSV 간 mt20Id 기준 diff 리포트

## 환경 셋업

```bash
sudo apt install -y python3.12-venv      # Debian/Ubuntu
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install playwright beautifulsoup4 requests
playwright install chromium
```

## 재현 흐름

### 1) 뮤지컬 mt20Id 목록 수집

```bash
.venv/bin/python enumerate_musicals.py \
  --out mt20ids_musical.txt \
  --page-size 100 --delay 0.3
```

### 2) detail 스크래핑 (백그라운드 권장, 약 14~18시간)

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

### 3) (필요 시) SKIP 항목 retry

본 추출 후 `skipped_musical.jsonl` 의 `not_a_record` / `genre_mismatch` 항목을 모아 한 번 더 돌리고, mt20Id 기준으로 본 CSV 와 병합. 자세한 절차는 `ai/tasks/common-workflows.md` 참고.

### 4) (선택) v1 과 diff

```bash
.venv/bin/python verify_rescrape.py \
  --old <old_csv> --new kopis_musical.csv \
  --out verify_report.md
```

## 과거 코드 보존

v1 (전체 PF iterate + 142k 행 CSV) 은 `legacy/v1-scraper` 브랜치(원격에도 있음) 에 그대로 보존.
