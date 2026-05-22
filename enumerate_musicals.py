"""KOPIS 공연DB 의 '뮤지컬' 카테고리 mt20Id를 JSON API 로 열거.

엔드포인트: https://kopis.or.kr:9001/api/prs/v1/por/db/prfrdb/perfo-infos
- tabno=ggga 가 뮤지컬 필터.
- 응답: {result: [...], prfrDbDTO: {...}}. result[i].prfrId 가 mt20Id, result[0].totcnt 가 총건수.

별도 인증/CSRF 없이 호출 가능하지만, 일부 환경에서 차단되면 Playwright 로
KOPIS 검색 페이지를 한 번 방문해 쿠키를 받아 requests.Session 에 주입한다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import List, Optional, Sequence

import requests

API_URL = "https://kopis.or.kr:9001/api/prs/v1/por/db/prfrdb/perfo-infos"
SEARCH_PAGE_URL = "https://kopis.or.kr/por/db/pblprfr/pblprfr.do?menuId=MNU_00020"
DEFAULT_TABNO = "ggga"  # 뮤지컬

DEFAULT_PARAMS = {
    "sPageIndex": "1",
    "pageRcdPer": "100",
    "orderGubun": "01",
    "tabno": DEFAULT_TABNO,
    "prfNm": "",
    "srchVisit": "",
    "signguCode": "",
    "signguCodeSub": "",
    "prfPdFrom": "",
    "prfPdTo": "",
    "prfState": "",
    "srchOpenRun": "",
    "mt2zGenreCode": "",
    "seatScale": "",
    "srchPrices": "",
    "menuGubun": "",
    "kidState": "",
    "festival": "",
    "fcltyChartr": "",
    "prfAwarded": "",
    "muscLicenAt": "",
    "muscCreatAt": "",
    "srchEtcs": "",
    "srchDt": "",
}

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/145.0 Safari/537.36"
)


def warm_cookies_via_playwright(session: requests.Session) -> None:
    # API가 차단할 경우 KOPIS 검색 페이지를 한 번 열어 쿠키를 가져온다.
    from playwright.sync_api import sync_playwright

    print("[WARM] launching playwright to seed cookies...", file=sys.stderr)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(locale="ko-KR")
        page = context.new_page()
        page.goto(SEARCH_PAGE_URL, wait_until="domcontentloaded", timeout=30000)
        for c in context.cookies():
            session.cookies.set(c["name"], c["value"], domain=c["domain"], path=c.get("path", "/"))
        browser.close()
    print(f"[WARM] cookies={len(session.cookies)}", file=sys.stderr)


def fetch_page(session: requests.Session, page_index: int, page_size: int, tabno: str) -> dict:
    params = dict(DEFAULT_PARAMS)
    params["sPageIndex"] = str(page_index)
    params["pageRcdPer"] = str(page_size)
    params["tabno"] = tabno
    r = session.get(
        API_URL,
        params=params,
        headers={"User-Agent": UA, "Accept": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="KOPIS 뮤지컬 mt20Id 열거")
    parser.add_argument("--out", default="mt20ids_musical.txt", help="결과 mt20Id 출력 경로")
    parser.add_argument("--log", default="enumerate_musicals.jsonl", help="페이지별 메타 로그(JSONL)")
    parser.add_argument("--page-size", type=int, default=100, help="페이지당 건수 (기본 100)")
    parser.add_argument("--delay", type=float, default=0.3, help="페이지 간 지연(초)")
    parser.add_argument("--tabno", default=DEFAULT_TABNO, help="장르 코드 (기본 ggga=뮤지컬)")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="안전상한(0=무제한). 디버그용으로 작게 지정 가능.",
    )
    args = parser.parse_args(argv)

    session = requests.Session()

    # 1) 첫 페이지 호출하여 총건수 확정. 차단되면 쿠키 시드 후 재시도.
    try:
        first = fetch_page(session, 1, args.page_size, args.tabno)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code in (401, 403, 419):
            warm_cookies_via_playwright(session)
            first = fetch_page(session, 1, args.page_size, args.tabno)
        else:
            raise

    items = first.get("result") or []
    if not items:
        print("ERROR: 첫 페이지 응답에 result 가 비어있음", file=sys.stderr)
        return 2
    totcnt = int(items[0].get("totcnt") or 0)
    if totcnt <= 0:
        print(f"ERROR: totcnt={totcnt} (응답 비정상)", file=sys.stderr)
        return 2
    total_pages = math.ceil(totcnt / args.page_size)
    if args.max_pages and total_pages > args.max_pages:
        total_pages = args.max_pages
    print(
        f"[INFO] totcnt={totcnt} page_size={args.page_size} total_pages={total_pages} tabno={args.tabno}"
    )

    # 2) 결과 파일은 새로 쓴다(idempotent). dedup 은 마지막에.
    all_ids: List[str] = []

    def write_log(record: dict) -> None:
        with open(args.log, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(record, ensure_ascii=False) + "\n")

    def absorb_page(page_idx: int, payload: dict) -> int:
        page_items = payload.get("result") or []
        ids_here = [it.get("prfrId") for it in page_items if it.get("prfrId")]
        first_id = ids_here[0] if ids_here else ""
        last_id = ids_here[-1] if ids_here else ""
        all_ids.extend(ids_here)
        write_log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "page": page_idx,
            "count": len(ids_here),
            "first_id": first_id,
            "last_id": last_id,
        })
        return len(ids_here)

    # 3) page=1 결과 흡수.
    n1 = absorb_page(1, first)
    print(f"[PAGE 1/{total_pages}] +{n1} (cum {len(all_ids)})")

    # 4) page=2..total_pages 순회.
    for p_idx in range(2, total_pages + 1):
        time.sleep(max(args.delay, 0.0))
        try:
            payload = fetch_page(session, p_idx, args.page_size, args.tabno)
        except requests.HTTPError as e:
            print(f"[PAGE {p_idx}] HTTP {e.response.status_code if e.response else '?'} — retry once", file=sys.stderr)
            time.sleep(2.0)
            payload = fetch_page(session, p_idx, args.page_size, args.tabno)
        n = absorb_page(p_idx, payload)
        print(f"[PAGE {p_idx}/{total_pages}] +{n} (cum {len(all_ids)})")
        if n == 0:
            print(f"[PAGE {p_idx}] empty result — 종료(데이터 끝)", file=sys.stderr)
            break

    # 5) dedup (페이지 경계에서 중복 가능성). 입력 순서는 유지(첫 등장 기준).
    seen = set()
    deduped: List[str] = []
    for mt in all_ids:
        if mt and mt not in seen:
            seen.add(mt)
            deduped.append(mt)

    # 6) 출력 저장.
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for mt in deduped:
            f.write(mt + "\n")
    os.replace(tmp, args.out)

    print(
        f"[DONE] api_returned={len(all_ids)} unique={len(deduped)} "
        f"expected_totcnt≈{totcnt} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
