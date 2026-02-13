# musical

KOPIS 공연 데이터 수집/정리 작업 공간.

## 구성
- `kopis_iterate_pf_playwright.py`: PF 범위 iterate 수집 (체크포인트/중간저장 지원)
- `kopis_to_csv_playwright.py`: mt20Id 목록 -> CSV 변환
- `recheck_genre_hint.py`: genre_hint 재추론
- `mt20ids_iterated.txt`: 성공한 mt20Id 누적
- `kopis_iterated.csv`: 수집 결과 CSV
- `kopis_iterate.checkpoint`: 이어받기 체크포인트

## 실행 예시
```bash
python kopis_iterate_pf_playwright.py --start 113846 --end 284000 --delay 0.3 --save-every 200 --resume
```

```bash
python recheck_genre_hint.py --input kopis_iterated.csv --delay 0.5 --log-every 10
```
