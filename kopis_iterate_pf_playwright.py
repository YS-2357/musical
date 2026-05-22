"""PF 아이디를 숫자 범위로 iterate 하며 유효한 상세 페이지만 수집.

주의:
- PF 번호가 연속적이라는 보장이 없어 실패 요청이 매우 많을 수 있습니다.
- 반드시 작은 범위로 먼저 테스트하세요.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from itertools import islice
from typing import Dict, Iterable, List, Optional, Sequence, Set
from urllib.parse import urlencode

from bs4 import BeautifulSoup

# Playwright 의존성은 파싱 함수만 쓸 때 불필요하므로 지연 import 한다.
# get_rendered_html / _probe_one / probe_ceiling / main 에서 필요 시 import.

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
FIELDNAMES = ["mt20Id", "url", "title", "genre_hint", "genre_norm", "is_musical", *KNOWN_LABELS]


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
    # 실측 결과 KOPIS 페이지는 <table>을 쓰지 않으므로 <dl>/<dt><dd>와
    # 텍스트 라인 스캔 폴백만 유지한다.
    result: Dict[str, str] = {}
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
    # 비상용 폴백: header 텍스트에 GENRE_HINTS 중 첫 매칭을 반환한다.
    # ("전체"는 의미가 약해 제외, multi-match 거부 룰은 제거.)
    # 정확한 장르는 extract_genre_badge() 가 결정론적으로 가져온다.
    for g in GENRE_HINTS:
        if g and g != "전체" and g in text:
            return g
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


def extract_genre_badge(soup: BeautifulSoup) -> Optional[str]:
    # KOPIS 상세 페이지는 장르를 <span class="DBDetail_cls_*"> 한 곳에 박아둔다.
    # 이 span 텍스트가 가장 신뢰할 만한 단일 신호다.
    el = soup.select_one('[class*="DBDetail_cls"]')
    if el:
        text = clean_text(el.get_text(" ", strip=True))
        if text and text != "전체":
            return text
    # 2차 폴백: <h4> 의 다음 형제 <span> 텍스트가 GENRE_HINTS와 정확 일치하면 사용.
    h4 = soup.select_one("h4")
    if h4:
        sib = h4.find_next_sibling("span")
        if sib:
            text = clean_text(sib.get_text(" ", strip=True))
            if text in GENRE_HINTS and text != "전체":
                return text
    # 3차 폴백: 라벨 기반(향후 KOPIS가 라벨 구조로 회귀할 가능성 대비).
    g = extract_genre_from_labels(soup)
    if g:
        return g
    # 4차 폴백: 헤더 텍스트 키워드 스캔(비상용).
    header_text = clean_text("\n".join(islice(soup.stripped_strings, 80)))
    return guess_genre_from_header(header_text)


def build_detail_url(mt20id: str) -> str:
    # mt20Id로 상세 페이지 URL을 만든다.
    query = urlencode({"menuId": "MNU_00020", "mt20Id": mt20id})
    return f"{BASE_DETAIL_URL}?{query}"


def looks_valid_record(title: Optional[str], data: Dict[str, str]) -> bool:
    # title 만 있어도 유효로 인정한다(과거 silent drop 방지).
    # title 이 비었거나 KOPIS 빈 페이지 마커면 무효.
    if not title:
        return False
    if "KOPIS | DB검색" in title:
        return False
    return True


def is_sparse_record(data: Dict[str, str]) -> bool:
    # 핵심 라벨이 하나도 없는 경우 sparse 로 분류해 skip-log 에 기록한다.
    return not any(k in data and data[k] for k in ("공연기간", "공연장소", "공연시간"))


def get_rendered_html(page, url: str, timeout_ms: int = 30000) -> str:
    # 브라우저로 실제 렌더링된 HTML을 가져온다.
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError  # noqa: F401

    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    # 장르 badge / 라벨 DOM이 나타날 때까지 대기(렌더링 지연 대응).
    # 가장 결정론적 신호인 DBDetail_cls 를 우선으로 둔다.
    for sel in (
        '[class*="DBDetail_cls"]',
        "dl dt",
        "text=공연기간",
        "text=공연장소",
        "text=공연시간",
    ):
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


def log_skip(
    path: str,
    reason: str,
    mt20id: str,
    url: str,
    snippet: str = "",
    http_title: str = "",
) -> None:
    # 스킵/이상 케이스 사유를 JSONL로 누적 기록한다 (silent drop 방지).
    if not path:
        return
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "mt20Id": mt20id,
        "url": url,
        "http_title": http_title,
        "snippet": snippet[:300] if snippet else "",
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def _probe_one(page, n: int) -> bool:
    # 단건 valid 여부 확인 (probe_ceiling 보조 함수).
    url = build_detail_url(f"PF{n:06d}")
    try:
        html = get_rendered_html(page, url)
        soup = BeautifulSoup(html, "html.parser")
        title = parse_title(soup)
        data = parse_label_value_blocks(soup)
        return looks_valid_record(title, data)
    except Exception:
        return False


def probe_ceiling(
    page,
    start_n: int,
    step: int = 500,
    max_skips: int = 10,
    max_consecutive_invalid: int = 200,
) -> int:
    # 1) coarse: start_n부터 step씩 점프, max_skips 연속 invalid면 종료.
    # 2) fine: last_valid 직후부터 1씩 증가, max_consecutive_invalid 연속 invalid면 종료.
    last_valid = start_n - 1
    skips_in_a_row = 0
    n = start_n
    print(f"[PROBE] coarse phase start={start_n} step={step} max_skips={max_skips}")
    while skips_in_a_row < max_skips:
        if _probe_one(page, n):
            last_valid = n
            skips_in_a_row = 0
            print(f"[PROBE coarse] valid PF{n:06d}")
        else:
            skips_in_a_row += 1
            print(f"[PROBE coarse] invalid PF{n:06d} streak={skips_in_a_row}/{max_skips}")
        n += step

    print(f"[PROBE] fine phase from PF{max(last_valid + 1, start_n):06d}")
    n = max(last_valid + 1, start_n)
    invalid_streak = 0
    while invalid_streak < max_consecutive_invalid:
        if _probe_one(page, n):
            last_valid = n
            invalid_streak = 0
            print(f"[PROBE fine] valid PF{n:06d}")
        else:
            invalid_streak += 1
        n += 1

    print(f"[PROBE] done last_valid=PF{last_valid:06d}")
    return last_valid


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PF 숫자 iterate 기반 수집기 (Playwright)")
    # 1) 실행 파라미터를 정의한다.
    parser.add_argument("--start", type=int, default=1, help="시작 숫자 (기본 1)")
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="끝 숫자 포함 (예: 285240). --auto-ceiling 이면 무시.",
    )
    parser.add_argument(
        "--auto-ceiling",
        action="store_true",
        help="probe_ceiling 으로 ceiling 자동탐지하여 --end 대신 사용. kopis_ceiling.txt에 저장.",
    )
    parser.add_argument(
        "--probe-start",
        type=int,
        default=280000,
        help="--auto-ceiling 시 probe 시작 PF 숫자 (기본 280000)",
    )
    parser.add_argument(
        "--ceiling-file",
        default="kopis_ceiling.txt",
        help="auto-ceiling 결과 저장 경로",
    )
    parser.add_argument("--delay", type=float, default=0.3, help="요청 간 지연(초, 기본 0.3)")
    parser.add_argument("--out-csv", default="kopis_iterated_v2.csv", help="출력 CSV")
    parser.add_argument("--out-ids", default="mt20ids_iterated_v2.txt", help="유효 mt20Id 출력")
    parser.add_argument(
        "--checkpoint",
        default="kopis_iterate_v2.checkpoint",
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
    parser.add_argument(
        "--skipped-log",
        default="skipped.jsonl",
        help="스킵/이상 케이스 사유를 누적 기록할 JSONL 파일",
    )

    args = parser.parse_args(argv)
    # 2) 입력 범위를 검증한다.
    if not args.auto_ceiling and args.end is None:
        parser.error("--end 또는 --auto-ceiling 중 하나는 필수입니다.")
    if args.end is not None and args.end < args.start:
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
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headful)
        context = browser.new_context(locale="ko-KR")
        page = context.new_page()

        # 5-1) --auto-ceiling 이면 본 루프 전에 ceiling 을 탐지한다.
        if args.auto_ceiling:
            ceiling = probe_ceiling(page, args.probe_start)
            with open(args.ceiling_file, "w", encoding="utf-8") as f:
                f.write(str(ceiling))
            print(f"[CEILING] PF{ceiling:06d} saved to {args.ceiling_file}")
            args.end = ceiling

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
                # 1) 장르 badge span(<span class="DBDetail_cls_*">)에서 결정론적 추출.
                #    실패 시 라벨/헤더-키워드 폴백.
                genre_norm = extract_genre_badge(soup) or ""
                is_musical = "True" if genre_norm == "뮤지컬" else "False"
                # genre_hint 는 구버전 컬럼 호환을 위해 genre_norm과 동일하게 유지.
                genre_hint = genre_norm
                data = parse_label_value_blocks(soup)
                # 6-3) 유효 레코드인지 검사 후, 새 성공건만 배치에 적재한다.
                if looks_valid_record(title, data):
                    row: Dict[str, str] = {
                        "mt20Id": mt20id,
                        "url": url,
                        "title": title or "",
                        "genre_hint": genre_hint,
                        "genre_norm": genre_norm,
                        "is_musical": is_musical,
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
                    # 핵심 라벨이 비어있으면 sparse 로 별도 기록(저장은 정상 진행).
                    if is_sparse_record(data):
                        log_skip(
                            args.skipped_log,
                            "valid_sparse",
                            mt20id,
                            url,
                            http_title=title or "",
                        )
                else:
                    # 유효하지 않으면 SKIP으로 기록한다.
                    skip_count += 1
                    print(f"[SKIP] {mt20id}")
                    snippet = clean_text("\n".join(islice(soup.stripped_strings, 80)))
                    if args.debug_skip:
                        print(f"[SKIP-DEBUG] {mt20id} snippet={snippet[:300]}")
                    log_skip(
                        args.skipped_log,
                        "not_a_record",
                        mt20id,
                        url,
                        snippet=snippet,
                        http_title=title or "",
                    )
            except Exception as e:  # noqa: BLE001
                # 예외는 ERR로 기록하고 다음 숫자로 진행한다.
                err_count += 1
                print(f"[ERR] {mt20id} {e}")
                log_skip(
                    args.skipped_log,
                    "render_error",
                    mt20id,
                    url,
                    snippet=str(e),
                )
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
