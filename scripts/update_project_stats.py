"""Update tracked public project statistics."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

DEFAULT_DATASET_ID = "contextecho2026/persona-drift-contextecho"
DEFAULT_HISTORICAL_DOWNLOADS = 39_000


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def fetch_hf_dataset_stats(dataset_id: str, timeout: int = 30) -> dict[str, Any]:
    url = f"https://huggingface.co/api/datasets/{dataset_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "ContextEcho-maintainer/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def month_key(value: str) -> str | None:
    if len(value) >= 7 and value[4] == "-":
        return value[:7]
    return None


def previous_month(period: str) -> str | None:
    if not month_key(period):
        return None
    year = int(period[:4])
    month = int(period[5:7])
    if month == 1:
        return f"{year - 1:04d}-12"
    return f"{year:04d}-{month - 1:02d}"


def download_buckets(current: dict[str, Any]) -> dict[str, int]:
    raw = current.get("dataset_hf_monthly_downloads")
    buckets: dict[str, int] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            count = as_int(value)
            if count is not None and count >= 0 and month_key(str(key)):
                buckets[str(key)[:7]] = count
    if buckets:
        return buckets

    previous_snapshot = as_int(current.get("dataset_hf_downloads_last_month_previous"))
    previous = as_int(current.get("dataset_hf_downloads_last_month"))
    previous_period = month_key(str(current.get("dataset_total_downloads_updated") or ""))
    if previous_snapshot is not None and previous_period:
        prior_period = previous_month(previous_period)
        if prior_period:
            buckets[prior_period] = previous_snapshot
    if previous is None:
        historical = as_int(current.get("dataset_historical_downloads")) or DEFAULT_HISTORICAL_DOWNLOADS
        total = as_int(current.get("dataset_total_downloads")) or historical
        inferred = total - historical
        previous = inferred if inferred >= 0 else None
    if previous is not None and previous_period and previous_period not in buckets:
        buckets[previous_period] = previous
    return buckets


def roll_download_total(current: dict[str, Any], hf_last_month: int | None, period: str) -> tuple[int, dict[str, int]]:
    historical = as_int(current.get("dataset_historical_downloads")) or DEFAULT_HISTORICAL_DOWNLOADS
    buckets = download_buckets(current)
    if hf_last_month is not None:
        existing = buckets.get(period)
        buckets[period] = max(existing or 0, hf_last_month)
    return historical + sum(buckets.values()), buckets


def update_stats(current: dict[str, Any], hf: dict[str, Any] | None, today: str) -> dict[str, Any]:
    out = dict(current)
    hf_last_month = as_int(hf.get("downloads")) if hf else as_int(current.get("dataset_hf_downloads_last_month"))
    period = month_key(today) or datetime.now(timezone.utc).strftime("%Y-%m")
    total, buckets = roll_download_total(out, hf_last_month, period)
    if hf_last_month is None:
        hf_last_month = buckets.get(period)
    out["dataset_historical_downloads"] = as_int(out.get("dataset_historical_downloads")) or DEFAULT_HISTORICAL_DOWNLOADS
    out["dataset_hf_monthly_downloads"] = dict(sorted(buckets.items()))
    out["dataset_hf_downloads_last_month"] = hf_last_month
    out["dataset_hf_downloads_last_month_period"] = period
    out.pop("dataset_hf_downloads_last_month_previous", None)
    out.pop("dataset_hf_downloads_last_month_delta_applied", None)
    out["dataset_total_downloads"] = total
    out["dataset_total_downloads_updated"] = today
    out["dataset_total_downloads_note"] = (
        "Maintainer-tracked cumulative download count: historical public totals "
        "plus one stored Hugging Face rolling last-month download bucket per month."
    )
    if hf:
        out["dataset_hf_likes"] = as_int(hf.get("likes"))
        if hf.get("lastModified"):
            out["dataset_hf_last_modified"] = hf.get("lastModified")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refresh tracked public project statistics.")
    p.add_argument("--stats", type=Path, default=Path("docs/project_stats.json"))
    p.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    p.add_argument("--check", action="store_true", help="fail if stats would change")
    p.add_argument("--allow-offline", action="store_true", help="preserve current stats if Hugging Face is unreachable")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    current = json.loads(args.stats.read_text(encoding="utf-8")) if args.stats.exists() else {}
    today = datetime.now(timezone.utc).date().isoformat()
    try:
        hf = fetch_hf_dataset_stats(args.dataset_id)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        if not args.allow_offline:
            print(f"[project-stats] failed to fetch Hugging Face stats: {exc}")
            return 1
        print(f"[project-stats] skipped Hugging Face stats: {exc}")
        hf = None
    rendered = json.dumps(update_stats(current, hf, today), indent=2) + "\n"
    if args.check:
        existing = args.stats.read_text(encoding="utf-8") if args.stats.exists() else ""
        if existing != rendered:
            print(f"[project-stats] stale: {args.stats}")
            return 1
        print(f"[project-stats] up to date: {args.stats}")
        return 0
    args.stats.write_text(rendered, encoding="utf-8")
    print(f"[project-stats] wrote {args.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
