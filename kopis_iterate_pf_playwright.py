"""PF 아이디를 숫자 범위로 iterate 하며 유효한 상세 페이지만 수집.

주의:
- PF 번호가 연속적이라는 보장이 없어 실패 요청이 매우 많을 수 있습니다.
- 반드시 작은 범위로 먼저 테스트하세요.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from itertools import islice
from typing import Dict, Iterable, List, Optional, Sequence, Set
from urllib.parse import urlencode

from bs4 import BeautifulSoup
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

BASE_DETAIL_URL = "https://kopis.or.kr/por/db/pblprfr/pblprfrView.do"

KNOWN_LABELS = [
    "공연기간",
    "공연장소",
    "공연시간",
    "관람연령",
    "티켓가격",
    "출연진",
    "창작자",
    "제작진",
    "주최·주관",
    "기획·제작",
    "최종수정",
]

LABEL_PATTERN = re.compile(r"^\s*({})\s*$".format("|".join(map(re.escape, KNOWN_LABELS))))
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
FIELDNAMES = ["mt20Id", "url", "title", "genre_hint", *KNOWN_LABELS]


def clean_text(s: str) -> str:
    # 공백/개행을 정규화해 파싱 안정성을 높인다.
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    return s.strip()


def normalize_value(value: str) -> str:
    # 라벨 값이 여러 줄일 때 단독 구분자 등을 제거하고 합친다.
    value = clean_text(value)
    if "\n" not in value:
        return value
    parts: List[str] = []
    for raw in value.split("\n"):
        part = clean_text(raw)
        if not part or part in {",", "·", "/"}:
            continue
        parts.append(part.strip(" ,"))
    return " / ".join(parts)


def parse_label_value_blocks(soup: BeautifulSoup) -> Dict[str, str]:
    # 상세 페이지의 라벨-값 구조를 최대한 폭넓게 수집한다.
    result: Dict[str, str] = {}
    for tr in soup.select("tr"):
        th = tr.find("th")
        td = tr.find("td")
        if not th or not td:
            continue
        label = clean_text(th.get_text(" ", strip=True))
        value = normalize_value(td.get_text("\n", strip=True))
        if label:
            result[label] = value
    for dt in soup.select("dt"):
        dd = dt.find_next_sibling("dd")
        if not dd:
            continue
        label = clean_text(dt.get_text(" ", strip=True))
        value = normalize_value(dd.get_text("\n", strip=True))
        if label and label not in result:
            result[label] = value
    if not result:
        lines = [clean_text(x) for x in soup.get_text("\n").split("\n")]
        lines = [x for x in lines if x]
        i = 0
        while i < len(lines):
            line = lines[i]
            if LABEL_PATTERN.match(line):
                label = line
                value_lines: List[str] = []
                j = i + 1
                while j < len(lines) and not LABEL_PATTERN.match(lines[j]):
                    value_lines.append(lines[j])
                    j += 1
                value = normalize_value("\n".join(v for v in value_lines if v))
                if label:
                    result[label] = value
                i = j
                continue
            i += 1
    if "최종수정" not in result:
        text = soup.get_text("\n")
        m = re.search(r"최종수정\s*[:：]\s*([0-9]{4}\.[0-9]{2}\.[0-9]{2})", text)
        if m:
            result["최종수정"] = m.group(1)
    return result


def parse_title(soup: BeautifulSoup) -> Optional[str]:
    # 다양한 제목 후보 셀렉터를 순서대로 시도한다.
    candidates = [
        soup.select_one("h2"),
        soup.select_one("h3"),
        soup.select_one("h4"),
        soup.select_one(".tit"),
        soup.select_one(".title"),
    ]
    for c in candidates:
        if c:
            t = clean_text(c.get_text(" ", strip=True))
            if t:
                t = re.sub(r"\s*뮤지컬\s*공유\s*$", "", t)
                return t
    lines = [clean_text(x) for x in soup.get_text("\n").split("\n")]
    lines = [x for x in lines if x]
    return lines[0] if lines else None


def guess_genre_from_header(text: str) -> Optional[str]:
    # 상단 텍스트에서 장르 힌트를 추정한다.
    # "전체"는 오탐이 많아 기본적으로 제외한다.
    matches: List[str] = []
    for genre in GENRE_HINTS:
        if genre and genre in text:
            matches.append(genre)
    # "전체"는 의미가 약하므로 제거
    matches = [m for m in matches if m != "전체"]
    # 복수 매칭이면 신뢰하기 어렵다고 보고 비움
    if len(matches) == 1:
        return matches[0]
    return None


def extract_genre_from_labels(soup: BeautifulSoup) -> Optional[str]:
    labels = parse_label_value_blocks(soup)
    for key in GENRE_LABELS:
        if key in labels:
            val = labels[key]
            for g in GENRE_HINTS:
                if g and g in val and g != "전체":
                    return g
    return None


def build_detail_url(mt20id: str) -> str:
    # mt20Id로 상세 페이지 URL을 만든다.
    query = urlencode({"menuId": "MNU_00020", "mt20Id": mt20id})
    return f"{BASE_DETAIL_URL}?{query}"


def looks_valid_record(title: Optional[str], data: Dict[str, str]) -> bool:
    # 공통/빈 페이지를 걸러내기 위한 최소 조건을 정의한다.
    if not title:
        return False
    if "KOPIS | DB검색" in title:
        return False
    # 핵심 라벨이 하나라도 있으면 유효로 간주
    if any(k in data and data[k] for k in ("공연기간", "공연장소", "공연시간")):
        return True
    # 구조가 바뀌었을 수 있으므로 라벨-값이 하나라도 있으면 유효로 간주
    if any(v for v in data.values()):
        return True
    return False


def get_rendered_html(page, url: str, timeout_ms: int = 30000) -> str:
    # 브라우저로 실제 렌더링된 HTML을 가져온다.
    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    # 핵심 라벨이 나타날 때까지 대기(렌더링 지연 대응)
    for sel in ("text=공연기간", "text=공연장소", "text=공연시간"):
        try:
            page.wait_for_selector(sel, timeout=5000)
            break
        except PlaywrightTimeoutError:
            continue
    page.wait_for_timeout(800)
    return page.content()


def union_fieldnames(rows: List[Dict[str, str]]) -> List[str]:
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


def _ensure_row_shape(row: Dict[str, str]) -> Dict[str, str]:
    # CSV 컬럼 순서/누락을 안정적으로 맞춘다.
    shaped: Dict[str, str] = {}
    for k in FIELDNAMES:
        shaped[k] = row.get(k, "")
    return shaped


def append_csv(path: str, rows: Iterable[Dict[str, str]]) -> None:
    # 중간 저장/이어받기를 위해 CSV에 append 방식으로 쓴다.
    rows = list(rows)
    if not rows:
        return
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(_ensure_row_shape(row))


def append_ids(path: str, ids: Iterable[str]) -> None:
    # 성공한 mt20Id만 별도 텍스트 파일에 누적 저장한다.
    ids = list(ids)
    if not ids:
        return
    with open(path, "a", encoding="utf-8") as f:
        for mt20id in ids:
            f.write(mt20id + "\n")


def load_existing_ids(path: str) -> Set[str]:
    # resume 시 중복 저장을 막기 위해 기존 성공 ID를 로드한다.
    if not os.path.exists(path):
        return set()
    existing: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            v = line.strip()
            if v:
                existing.add(v)
    return existing


def load_checkpoint(path: str) -> Optional[int]:
    # 마지막으로 시도한 숫자를 읽어 이어받기 시작점을 결정한다.
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            v = f.read().strip()
        return int(v) if v else None
    except Exception:
        return None


def save_checkpoint(path: str, n: int) -> None:
    # 현재까지 진행한 숫자를 기록해 중단 시 이어받기 가능하게 한다.
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(n))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PF 숫자 iterate 기반 수집기 (Playwright)")
    # 1) 실행 파라미터를 정의한다.
    parser.add_argument("--start", type=int, required=True, help="시작 숫자 (예: 280000)")
    parser.add_argument("--end", type=int, required=True, help="끝 숫자 포함 (예: 284000)")
    parser.add_argument("--delay", type=float, default=0.15, help="요청 간 지연(초)")
    parser.add_argument("--out-csv", default="kopis_iterated.csv", help="출력 CSV")
    parser.add_argument("--out-ids", default="mt20ids_iterated.txt", help="유효 mt20Id 출력")
    parser.add_argument(
        "--checkpoint",
        default="kopis_iterate.checkpoint",
        help="이어받기용 체크포인트 파일",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="체크포인트/기존 ids를 읽어 이어서 진행",
    )
    parser.add_argument(
        "--ignore-seen",
        action="store_true",
        help="기존 ids 목록을 무시하고 다시 저장(중복 위험)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=200,
        help="이 시도 횟수마다 중간 저장/체크포인트 기록",
    )
    parser.add_argument(
        "--debug-skip",
        action="store_true",
        help="SKIP 발생 시 페이지 텍스트 일부를 출력",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="이 시도 횟수마다 진행 로그 출력(기본 50)",
    )
    parser.add_argument("--headful", action="store_true", help="브라우저 표시")

    args = parser.parse_args(argv)
    # 2) 입력 범위를 검증한다.
    if args.end < args.start:
        parser.error("--end는 --start 이상이어야 합니다.")

    # 3) resume 옵션이면 시작 지점과 기존 성공 ID를 복원한다.
    start_n = args.start
    seen_ids: Set[str] = set()
    if args.resume:
        ck = load_checkpoint(args.checkpoint)
        if ck is not None:
            start_n = max(start_n, ck + 1)
            print(f"[RESUME] checkpoint={ck} -> start={start_n}")
        if not args.ignore_seen:
            seen_ids = load_existing_ids(args.out_ids)
            if seen_ids:
                print(f"[RESUME] 기존 ids 로드: {len(seen_ids)}개")

    # 4) 중간 저장을 위한 배치 버퍼를 준비한다.
    batch_rows: List[Dict[str, str]] = []
    batch_ids: List[str] = []

    # 5) Playwright 브라우저를 열고 단일 페이지로 순회한다.
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headful)
        context = browser.new_context(locale="ko-KR")
        page = context.new_page()

        # 6) PF 숫자 범위를 순회하며 상세 페이지를 파싱한다.
        attempts = 0
        ok_new = 0
        skip_count = 0
        err_count = 0
        for n in range(start_n, args.end + 1):
            attempts += 1
            # 6-1) 현재 숫자를 mt20Id(PFxxxxxx) 형태로 만든다.
            mt20id = f"PF{n:06d}"
            url = build_detail_url(mt20id)
            try:
                # 6-2) 상세 페이지를 로드하고 파싱 가능한 형태로 만든다.
                html = get_rendered_html(page, url)
                soup = BeautifulSoup(html, "html.parser")
                title = parse_title(soup)
                # 1) 라벨 기반 장르 추론(가장 정확)
                genre_hint = extract_genre_from_labels(soup)
                # 2) 실패 시 상단 텍스트 기반 추론
                if not genre_hint:
                    header_text = clean_text("\n".join(islice(soup.stripped_strings, 80)))
                    genre_hint = guess_genre_from_header(header_text)
                data = parse_label_value_blocks(soup)
                # 6-3) 유효 레코드인지 검사 후, 새 성공건만 배치에 적재한다.
                if looks_valid_record(title, data):
                    row: Dict[str, str] = {
                        "mt20Id": mt20id,
                        "url": url,
                        "title": title or "",
                        "genre_hint": genre_hint or "",
                    }
                    row.update(data)
                    if mt20id not in seen_ids:
                        batch_rows.append(row)
                        batch_ids.append(mt20id)
                        seen_ids.add(mt20id)
                        ok_new += 1
                        print(f"[OK] {mt20id} {title}")
                    else:
                        print(f"[OK-EXIST] {mt20id} {title}")
                else:
                    # 유효하지 않으면 SKIP으로 기록한다.
                    skip_count += 1
                    print(f"[SKIP] {mt20id}")
                    if args.debug_skip:
                        snippet = clean_text("\n".join(islice(soup.stripped_strings, 80)))
                        print(f"[SKIP-DEBUG] {mt20id} snippet={snippet[:300]}")
            except Exception as e:  # noqa: BLE001
                # 예외는 ERR로 기록하고 다음 숫자로 진행한다.
                err_count += 1
                print(f"[ERR] {mt20id} {e}")
            # 6-4) 서버 부하/차단 완화를 위해 지연을 둔다.
            time.sleep(max(args.delay, 0.0))

            # 7) 진행 상황을 주기적으로 출력한다.
            if args.log_every > 0 and attempts % args.log_every == 0:
                print(
                    f"[PROGRESS] at={n} attempts={attempts} ok_new={ok_new} "
                    f"skip={skip_count} err={err_count} total_seen={len(seen_ids)}"
                )

            # 8) 주기적으로 중간 저장 + 체크포인트를 남긴다.
            if args.save_every > 0 and attempts % args.save_every == 0:
                append_csv(args.out_csv, batch_rows)
                append_ids(args.out_ids, batch_ids)
                save_checkpoint(args.checkpoint, n)
                print(
                    f"[SAVE] at={n} batch_ids={len(batch_ids)} total_seen={len(seen_ids)}",
                    file=sys.stderr,
                )
                batch_rows.clear()
                batch_ids.clear()

        context.close()
        browser.close()

    # 9) 루프 종료 후 남은 배치를 저장하고 체크포인트를 갱신한다.
    append_csv(args.out_csv, batch_rows)
    append_ids(args.out_ids, batch_ids)
    save_checkpoint(args.checkpoint, args.end)

    # 10) 최종 요약을 출력하고 종료한다.
    print(
        f"완료: total_ids={len(seen_ids)} csv={args.out_csv} ids={args.out_ids} checkpoint={args.checkpoint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
