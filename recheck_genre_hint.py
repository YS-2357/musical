"""genre_hint가 '전체'인 행만 다시 확인해서 갱신.

- 입력 CSV의 URL을 다시 방문해 header 텍스트에서 장르를 추정
- 새 장르를 찾으면 덮어쓰기, 못 찾으면 빈 값으로 설정

사용 예시:
  python .\recheck_genre_hint.py --input .\kopis_iterated.csv --delay 0.2 --log-every 10
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from itertools import islice
from typing import Dict, Iterable, List, Optional

from bs4 import BeautifulSoup
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

GENRE_HINTS = [
    "연극",
    "뮤지컬",
    "서양음악(클래식)",
    "한국음악(국악)",
    "대중음악",
    "무용(서양/한국무용)",
    "대중무용",
    "서커스/마술",
    "복합",
    "전체",
]
GENRE_LABELS = [
    "장르",
    "공연장르",
    "공연분야",
    "분야",
    "분류",
]


def clean_text(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    return s.strip()


def guess_genre_from_header(text: str) -> Optional[str]:
    # "전체"는 오탐이 많으므로 제거, 복수 매칭이면 None 처리
    matches: List[str] = []
    for genre in GENRE_HINTS:
        if genre and genre in text:
            matches.append(genre)
    matches = [m for m in matches if m != "전체"]
    if len(matches) == 1:
        return matches[0]
    return None


def parse_label_value_blocks(soup: BeautifulSoup) -> Dict[str, str]:
    result: Dict[str, str] = {}
    # table 구조(th/td)
    for tr in soup.select("tr"):
        th = tr.find("th")
        td = tr.find("td")
        if not th or not td:
            continue
        label = clean_text(th.get_text(" ", strip=True))
        value = clean_text(td.get_text("\n", strip=True))
        if label:
            result[label] = value
    # 정의 리스트(dt/dd)
    for dt in soup.select("dt"):
        dd = dt.find_next_sibling("dd")
        if not dd:
            continue
        label = clean_text(dt.get_text(" ", strip=True))
        value = clean_text(dd.get_text("\n", strip=True))
        if label and label not in result:
            result[label] = value
    return result


def extract_genre_from_labels(soup: BeautifulSoup) -> Optional[str]:
    labels = parse_label_value_blocks(soup)
    for key in GENRE_LABELS:
        if key in labels:
            val = labels[key]
            # 라벨 값에 GENRE_HINTS가 포함되면 그 값을 우선
            for g in GENRE_HINTS:
                if g and g in val:
                    return g
    return None


def get_rendered_html(page, url: str, timeout_ms: int = 30000) -> str:
    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    try:
        page.wait_for_selector("text=공연기간", timeout=4000)
    except PlaywrightTimeoutError:
        pass
    page.wait_for_timeout(600)
    return page.content()


def iter_rows(path: str) -> Iterable[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="genre_hint='전체' 재확인 스크립트")
    parser.add_argument("--input", required=True, help="입력 CSV(동일 파일로 덮어씀)")
    parser.add_argument("--output", help="출력 CSV(미지정 시 입력 파일 덮어쓰기)")
    parser.add_argument("--delay", type=float, default=0.5, help="요청 간 지연(초)")
    parser.add_argument("--max", type=int, default=0, help="최대 처리 건수(0=무제한)")
    parser.add_argument("--headful", action="store_true", help="브라우저 표시")
    parser.add_argument("--log-every", type=int, default=10, help="진행 로그 간격")
    args = parser.parse_args(argv)

    rows_iter = list(iter_rows(args.input))
    out_path = args.output or args.input
    tmp_path = out_path + ".tmp"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headful)
        context = browser.new_context(locale="ko-KR")
        page = context.new_page()

        processed = 0
        updated = 0
        checked = 0

        with open(tmp_path, "w", encoding="utf-8-sig", newline="") as f:
            writer: Optional[csv.DictWriter] = None

            for row in rows_iter:
                processed += 1
                if args.max and processed > args.max:
                    break

                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    writer.writeheader()

                current_hint = (row.get("genre_hint", "") or "").strip()
                if current_hint == "":
                    checked += 1
                    url = row.get("url", "")
                    if url:
                        try:
                            html = get_rendered_html(page, url)
                            soup = BeautifulSoup(html, "html.parser")
                            # 1) 라벨 기반(가장 정확)
                            new_genre = extract_genre_from_labels(soup)
                            # 2) 실패 시 상단 텍스트 기반
                            if not new_genre:
                                header_text = clean_text(
                                    "\n".join(islice(soup.stripped_strings, 80))
                                )
                                new_genre = guess_genre_from_header(header_text)
                            row["genre_hint"] = new_genre or ""
                            updated += 1
                            print(f"[UPDATE] {row.get('mt20Id','')} -> {row['genre_hint']}")
                        except Exception as e:  # noqa: BLE001
                            print(f"[ERR] {row.get('mt20Id','')} {e}", file=sys.stderr)
                    else:
                        row["genre_hint"] = ""

                writer.writerow(row)
                if args.log_every > 0 and processed % args.log_every == 0:
                    print(
                        f"[PROGRESS] processed={processed} checked={checked} updated={updated}"
                    )
                time.sleep(max(args.delay, 0.0))

        context.close()
        browser.close()

    # 임시 파일을 최종 파일로 교체 (원본 보호)
    try:
        os.replace(tmp_path, out_path)
    except Exception as e:  # noqa: BLE001
        print(f"[ERR] 파일 교체 실패: {e}", file=sys.stderr)
        print(f"[INFO] 임시 파일 보존: {tmp_path}", file=sys.stderr)
        return 2

    print(f"완료: processed={processed} updated={updated} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
