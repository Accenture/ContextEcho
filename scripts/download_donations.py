"""Download private Hugging Face staging donations for maintainer review."""
from __future__ import annotations

import argparse
from pathlib import Path

STAGING_REPO = "contextecho2026/persona-drift-staging"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download ContextEcho staged donations.")
    p.add_argument("--repo-id", default=STAGING_REPO)
    p.add_argument("--repo-type", default="dataset")
    p.add_argument("--local-dir", type=Path, default=Path("hf_staging_download"))
    p.add_argument("--token", default=None, help="HF token; defaults to HF_TOKEN/cached login")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[error] huggingface_hub is not installed. Run `make setup-donate`.")
        return 2

    out = snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        local_dir=str(args.local_dir),
        token=args.token,
    )
    pending = sorted((Path(out) / "pending").glob("submission-*"))
    print(f"[download] repo      : {args.repo_id}")
    print(f"[download] local dir : {Path(out).resolve()}")
    print(f"[download] submissions: {len(pending)}")
    for sub in pending:
        print(f"  - {sub}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
