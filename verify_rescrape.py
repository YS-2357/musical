"""kopis_iterated.csv(구) 와 kopis_iterated_v2.csv(신) 의 diff 리포트.

mt20Id 기준 outer join 후 verify_report.md 를 생성한다.
- 신측 is_musical=True 카운트 vs 구측 genre_hint=뮤지컬 카운트
- 새로 정확히 분류된 뮤지컬 / 회귀(이전 뮤지컬 -> 신측 다른 장르) 목록
- 컬럼별 non-empty 비율
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from typing import Dict, List, Optional


def load_csv(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return {row["mt20Id"]: row for row in reader if row.get("mt20Id")}


def non_empty_ratio(rows: List[Dict[str, str]], col: str) -> float:
    if not rows:
        return 0.0
    total = len(rows)
    n = sum(1 for r in rows if (r.get(col) or "").strip() not in ("", "해당정보 없음"))
    return n / total


def is_musical_old(row: Dict[str, str]) -> bool:
    return (row.get("genre_hint") or "").strip() == "뮤지컬"


def is_musical_new(row: Dict[str, str]) -> bool:
    if (row.get("is_musical") or "").strip() == "True":
        return True
    return (row.get("genre_norm") or row.get("genre_hint") or "").strip() == "뮤지컬"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="신구 KOPIS CSV diff 리포트")
    parser.add_argument("--old", default="kopis_iterated.csv")
    parser.add_argument("--new", default="kopis_iterated_v2.csv")
    parser.add_argument("--out", default="verify_report.md")
    parser.add_argument(
        "--list-limit",
        type=int,
        default=50,
        help="새 뮤지컬/회귀 목록을 최대 몇 건까지 리포트에 직접 나열할지",
    )
    args = parser.parse_args(argv)

    old = load_csv(args.old)
    new = load_csv(args.new)

    old_ids = set(old.keys())
    new_ids = set(new.keys())
    common = old_ids & new_ids
    only_old = old_ids - new_ids
    only_new = new_ids - old_ids

    old_musical = {mt for mt in old_ids if is_musical_old(old[mt])}
    new_musical = {mt for mt in new_ids if is_musical_new(new[mt])}

    new_musical_in_common = {mt for mt in common if is_musical_new(new[mt])}
    old_musical_in_common = {mt for mt in common if is_musical_old(old[mt])}

    newly_classified_musical = sorted(new_musical_in_common - old_musical_in_common)
    regressed_musical = sorted(old_musical_in_common - new_musical_in_common)

    cols_to_check = [
        "title",
        "genre_hint",
        "genre_norm",
        "is_musical",
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
    old_rows = list(old.values())
    new_rows = list(new.values())

    new_genre_dist = Counter((r.get("genre_norm") or r.get("genre_hint") or "").strip() for r in new_rows)
    old_genre_dist = Counter((r.get("genre_hint") or "").strip() for r in old_rows)

    lines: List[str] = []
    lines.append("# KOPIS rescrape verify report")
    lines.append("")
    lines.append(f"- old: `{args.old}` rows={len(old_rows)}")
    lines.append(f"- new: `{args.new}` rows={len(new_rows)}")
    lines.append(f"- common mt20Ids: {len(common)}")
    lines.append(f"- only in old: {len(only_old)}")
    lines.append(f"- only in new: {len(only_new)}")
    lines.append("")
    lines.append("## Musical counts")
    lines.append(f"- old (genre_hint == 뮤지컬): {len(old_musical)}")
    lines.append(f"- new (is_musical == True or genre_norm == 뮤지컬): {len(new_musical)}")
    lines.append(f"- newly-classified musicals (in common, old-not-musical → new-musical): {len(newly_classified_musical)}")
    lines.append(f"- regressed musicals (in common, old-musical → new-not-musical): {len(regressed_musical)}")
    lines.append("")
    lines.append("## Genre distribution")
    lines.append("| genre | old | new |")
    lines.append("|---|---:|---:|")
    all_keys = sorted(set(old_genre_dist) | set(new_genre_dist), key=lambda k: -(new_genre_dist.get(k, 0)))
    for k in all_keys:
        lines.append(f"| {k or '(empty)'} | {old_genre_dist.get(k, 0)} | {new_genre_dist.get(k, 0)} |")
    lines.append("")
    lines.append("## Column non-empty ratio")
    lines.append("| column | old | new |")
    lines.append("|---|---:|---:|")
    for col in cols_to_check:
        lines.append(f"| {col} | {non_empty_ratio(old_rows, col):.3f} | {non_empty_ratio(new_rows, col):.3f} |")
    lines.append("")
    lines.append(f"## Newly classified musicals (top {args.list_limit})")
    for mt in newly_classified_musical[: args.list_limit]:
        title = new[mt].get("title", "")
        old_g = old[mt].get("genre_hint", "")
        new_g = new[mt].get("genre_norm", new[mt].get("genre_hint", ""))
        lines.append(f"- {mt}: `{old_g}` → `{new_g}` — {title}")
    if len(newly_classified_musical) > args.list_limit:
        lines.append(f"- ... ({len(newly_classified_musical) - args.list_limit} more)")
    lines.append("")
    lines.append(f"## Regressed musicals (top {args.list_limit})")
    for mt in regressed_musical[: args.list_limit]:
        title = new[mt].get("title", "")
        new_g = new[mt].get("genre_norm", new[mt].get("genre_hint", ""))
        lines.append(f"- {mt}: 뮤지컬 → `{new_g}` — {title}")
    if len(regressed_musical) > args.list_limit:
        lines.append(f"- ... ({len(regressed_musical) - args.list_limit} more)")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
