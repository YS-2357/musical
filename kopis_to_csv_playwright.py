"""KOPIS 공연 상세 페이지를 Playwright로 수집해 CSV로 저장.

requests로는 상세 데이터가 비어오는 경우가 있어 브라우저 렌더링을 사용합니다.

사용 예시:
  python3 kopis_to_csv_playwright.py --urls-file kopis_urls.txt --output kopis_pw_test.csv --delay 0.8
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass
from itertools import islice
from typing import Dict, Iterable, List, Optional
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
    "전체",
    "연극",
    "뮤지컬",
    "서양음악(클래식)",
    "한국음악(국악)",
    "대중음악",
    "무용(서양/한국무용)",
    "대중무용",
    "서커스/마술",
    "복합",
]


@dataclass
class PerformanceRecord:
    url: str
    mt20Id: Optional[str]
    title: Optional[str]
    genre_hint: Optional[str]
    last_updated: Optional[str]
    data: Dict[str, str]
    error: Optional[str] = None

    def to_flat_dict(self) -> Dict[str, str]:
        flat: Dict[str, str] = {
            "url": self.url,
            "mt20Id": self.mt20Id or "",
            "title": self.title or "",
            "genre_hint": self.genre_hint or "",
            "last_updated": self.last_updated or "",
            "error": self.error or "",
        }
        for k, v in self.data.items():
            flat[k] = v
        return flat


def build_detail_url(mt20id: str) -> str:
    query = urlencode({"menuId": "MNU_00020", "mt20Id": mt20id})
    return f"{BASE_DETAIL_URL}?{query}"


def extract_mt20id(url: str) -> Optional[str]:
    m = re.search(r"[?&]mt20Id=([A-Z0-9]+)", url)
    return m.group(1) if m else None


def extract_mt10id_from_text(text: str) -> Optional[str]:
    m = re.search(r"[?&]mt10Id=([A-Z0-9]+)", text)
    return m.group(1) if m else None


def extract_mt10id_from_soup(soup: BeautifulSoup) -> Optional[str]:
    # 공연장 링크 등에서 mt10Id를 추출
    for a in soup.select("a[href*='mt10Id=']"):
        href = a.get("href") or ""
        mt10id = extract_mt10id_from_text(href)
        if mt10id:
            return mt10id
    return None


def clean_text(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    return s.strip()


def normalize_value(value: str) -> str:
    """라벨 값의 줄바꿈/단독 쉼표 등을 사람이 읽기 좋은 형태로 정리."""
    value = clean_text(value)
    if "\n" not in value:
        return value

    parts: List[str] = []
    for raw in value.split("\n"):
        part = clean_text(raw)
        # 단독 쉼표나 구분자 역할의 잔여 문자 제거
        if not part or part in {",", "·", "/"}:
            continue
        parts.append(part.strip(" ,"))

    return " / ".join(parts)


def guess_genre_from_header(text: str) -> Optional[str]:
    for genre in GENRE_HINTS:
        if genre and genre in text:
            return genre
    return None


def parse_bls_tbbk_tables(soup: BeautifulSoup) -> Dict[str, str]:
    """공연장 탭 등에서 쓰이는 .bls_tbbk 테이블을 파싱."""
    result: Dict[str, str] = {}
    for table in soup.select(".bls_tbbk"):
        for tr in table.select("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if not th or not td:
                continue
            label = clean_text(th.get_text(" ", strip=True))
            value = normalize_value(td.get_text("\n", strip=True))
            if label:
                result[label] = value
    return result


def build_venue_url(mt10id: str) -> str:
    query = urlencode({"menuId": "MNU_00020", "mt10Id": mt10id})
    return f"{BASE_DETAIL_URL}?{query}"


def scrape_venue_via_click(page) -> Dict[str, str]:
    """페이지 내 공연장 링크를 클릭해 공연장 상세(.bls_tbbk)를 파싱."""
    try:
        venue_links = page.locator("a[href*='mt10Id=']")
        if venue_links.count() == 0:
            return {}
        venue_links.first.click(timeout=5000)
        try:
            page.wait_for_selector(".bls_tbbk", timeout=7000)
        except PlaywrightTimeoutError:
            page.wait_for_timeout(800)
        venue_html = page.content()
        venue_soup = BeautifulSoup(venue_html, "html.parser")
        return parse_bls_tbbk_tables(venue_soup)
    except Exception:
        return {}


def parse_label_value_blocks(soup: BeautifulSoup) -> Dict[str, str]:
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


def get_rendered_html(page, url: str, timeout_ms: int = 30000) -> str:
    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

    # 라벨 텍스트가 등장할 때까지 잠깐 대기 (없어도 진행)
    try:
        page.wait_for_selector("text=공연기간", timeout=5000)
    except PlaywrightTimeoutError:
        pass

    # 공연장 탭을 눌러 .bls_tbbk 정보가 로드되도록 시도
    try:
        page.locator("text=공연장").first.click(timeout=3000)
        page.wait_for_selector(".bls_tbbk", timeout=5000)
    except Exception:
        # 탭 클릭이 실패해도 본문 파싱은 계속 진행
        pass

    # 동적 렌더링 여지를 조금 더 줌
    page.wait_for_timeout(800)
    return page.content()


def scrape_detail_with_page(page, url: str) -> PerformanceRecord:
    mt20id = extract_mt20id(url)
    try:
        html = get_rendered_html(page, url)
        soup = BeautifulSoup(html, "html.parser")

        title = parse_title(soup)
        header_text = clean_text("\n".join(islice(soup.stripped_strings, 40)))
        genre_hint = guess_genre_from_header(header_text)

        label_values = parse_label_value_blocks(soup)
        # 공연장 탭의 .bls_tbbk 정보를 병합 (필수 필드 보강)
        bls_tbbk_values = parse_bls_tbbk_tables(soup)
        for k, v in bls_tbbk_values.items():
            if k not in label_values or not label_values.get(k):
                label_values[k] = v

        # mt10Id(공연장 ID)를 찾을 수 있으면 공연장 상세 페이지도 시도
        mt10id = extract_mt10id_from_soup(soup) or extract_mt10id_from_text(html)
        if mt10id:
            venue_url = build_venue_url(mt10id)
            try:
                venue_html = get_rendered_html(page, venue_url)
                venue_soup = BeautifulSoup(venue_html, "html.parser")
                venue_values = parse_bls_tbbk_tables(venue_soup)
                for k, v in venue_values.items():
                    if k not in label_values or not label_values.get(k):
                        label_values[k] = v
            except Exception:
                # 공연장 상세 이동이 실패해도 본문 결과는 유지
                pass
            finally:
                # 다음 루프를 위해 원래 공연 상세로 복귀
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                except Exception:
                    pass
        else:
            # mt10Id를 못 찾았으면 공연장 링크 클릭으로 시도
            venue_values = scrape_venue_via_click(page)
            for k, v in venue_values.items():
                if k not in label_values or not label_values.get(k):
                    label_values[k] = v
            # 클릭으로 이동했을 수 있으니 원래 공연 상세로 복귀
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
            except Exception:
                pass
        last_updated = label_values.get("최종수정")

        return PerformanceRecord(
            url=url,
            mt20Id=mt20id,
            title=title,
            genre_hint=genre_hint,
            last_updated=last_updated,
            data=label_values,
        )
    except Exception as e:  # noqa: BLE001
        return PerformanceRecord(
            url=url,
            mt20Id=mt20id,
            title=None,
            genre_hint=None,
            last_updated=None,
            data={},
            error=str(e),
        )


def iter_urls_from_file(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if not url or url.startswith("#"):
                continue
            yield url


def iter_urls_from_mt20ids_file(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            mt20id = line.strip()
            if not mt20id or mt20id.startswith("#"):
                continue
            yield build_detail_url(mt20id)


def union_fieldnames(rows: List[Dict[str, str]]) -> List[str]:
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


def write_csv(path: str, rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise ValueError("rows is empty")
    fieldnames = union_fieldnames(rows)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="KOPIS 상세 페이지 -> CSV (Playwright)")
    parser.add_argument("--urls-file", help="상세 페이지 URL 목록 파일")
    parser.add_argument("--mt20ids-file", help="mt20Id 목록 파일")
    parser.add_argument("--output", default="kopis_playwright.csv", help="출력 CSV 경로")
    parser.add_argument("--delay", type=float, default=0.7, help="요청 간 지연(초)")
    parser.add_argument("--max", type=int, default=0, help="최대 수집 건수(0=무제한)")
    parser.add_argument("--headful", action="store_true", help="브라우저를 표시(headed) 모드로 실행")

    args = parser.parse_args(argv)

    if not args.urls_file and not args.mt20ids_file:
        parser.error("--urls-file 또는 --mt20ids-file 중 하나는 필요합니다.")

    if args.urls_file and args.mt20ids_file:
        parser.error("--urls-file과 --mt20ids-file은 동시에 사용할 수 없습니다.")

    if args.urls_file:
        url_iter = iter_urls_from_file(args.urls_file)
    else:
        url_iter = iter_urls_from_mt20ids_file(args.mt20ids_file)

    rows: List[Dict[str, str]] = []
    total = 0
    errors = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headful)
        context = browser.new_context(locale="ko-KR")
        page = context.new_page()

        for url in url_iter:
            total += 1
            if args.max and total > args.max:
                break

            rec = scrape_detail_with_page(page, url)
            if rec.error:
                errors += 1
                print(f"[WARN] 실패: {url} -> {rec.error}", file=sys.stderr)
            else:
                print(f"[OK] {rec.title or ''} ({rec.mt20Id or ''})")

            rows.append(rec.to_flat_dict())
            time.sleep(max(args.delay, 0.0))

        context.close()
        browser.close()

    if not rows:
        print("수집된 데이터가 없습니다.", file=sys.stderr)
        return 2

    write_csv(args.output, rows)
    print(
        f"완료: total={len(rows)} errors={errors} output={args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
