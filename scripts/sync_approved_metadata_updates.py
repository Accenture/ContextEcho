"""Apply approved relay metadata updates to the local release archive."""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_RELAY_URL = "https://contextecho2026-context-echo-donation-relay.hf.space"
DEFAULT_DATASET_ROOT = Path("data_archive_release_v2")
PATCH_KEYS = ("credit_name", "contributor_email", "contributor_institute", "public_anonymous")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def metadata_patch(update: dict[str, Any]) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    for key in PATCH_KEYS:
        if key == "public_anonymous":
            if key in update:
                patch[key] = bool(update.get(key))
            continue
        value = str(update.get(key) or "").strip()
        if value:
            patch[key] = value
    return patch


def fetch_metadata_updates(relay_url: str, admin_token: str, limit: int) -> list[dict[str, Any]]:
    url = f"{relay_url.rstrip('/')}/api/admin/metadata-updates?limit={limit}"
    req = urllib.request.Request(url, headers={"X-Admin-Token": admin_token})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"relay metadata update fetch failed: HTTP {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"relay metadata update fetch failed: {exc}") from exc
    return list(payload.get("requests") or [])


def apply_approved_updates(dataset_root: Path, updates: list[dict[str, Any]]) -> dict[str, int]:
    ledger_path = dataset_root / "data" / "donations" / "ledger.jsonl"
    rows = read_jsonl(ledger_path)
    changed_ledger = 0
    changed_manifests = 0
    skipped = 0

    by_submission = {str(row.get("submission_id") or ""): row for row in rows}
    for update in updates:
        if update.get("status") != "approved":
            continue
        submission_id = str(update.get("submission_id") or "").strip()
        row = by_submission.get(submission_id)
        patch = metadata_patch(update)
        if not row or not patch:
            skipped += 1
            continue

        before = dict(row)
        for key, value in patch.items():
            if key == "contributor_institute":
                row["institute"] = value
            row[key] = value
        if row != before:
            changed_ledger += 1

        manifest_rel = str(row.get("manifest_path") or "").strip()
        manifest_path = dataset_root / manifest_rel if manifest_rel else Path()
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            before_manifest = dict(manifest)
            manifest.update(patch)
            manifest["metadata_update_request_id"] = str(update.get("request_id") or "")
            approved_utc = str(update.get("approved_utc") or "").strip()
            if approved_utc:
                manifest["metadata_updated_utc"] = approved_utc
            if manifest != before_manifest:
                write_json(manifest_path, manifest)
                changed_manifests += 1

    if changed_ledger:
        write_jsonl(ledger_path, rows)
    return {"ledger": changed_ledger, "manifests": changed_manifests, "skipped": skipped}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sync approved relay metadata updates into the local release archive.")
    p.add_argument("--relay-url", default=os.environ.get("CONTEXTECHO_RELAY_URL", DEFAULT_RELAY_URL))
    p.add_argument("--admin-token", default=os.environ.get("CONTEXTECHO_RELAY_ADMIN_TOKEN", ""))
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--limit", type=int, default=1000)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.admin_token:
        print("[metadata-sync] skipped: CONTEXTECHO_RELAY_ADMIN_TOKEN is not set")
        return 0
    ledger = args.dataset_root / "data" / "donations" / "ledger.jsonl"
    if not ledger.exists():
        print(f"[metadata-sync] skipped: local ledger not found at {ledger}")
        return 0
    updates = fetch_metadata_updates(args.relay_url, args.admin_token, args.limit)
    result = apply_approved_updates(args.dataset_root, updates)
    print(
        "[metadata-sync] applied approved updates: "
        f"ledger={result['ledger']} manifests={result['manifests']} skipped={result['skipped']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
