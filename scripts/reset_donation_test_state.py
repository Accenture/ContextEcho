"""Reset maintainer-local donation test state after archiving it.

This is intentionally local-only. It does not delete private Hugging Face
staging submissions and it does not touch donor-side Downloads folders.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return True


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def confirm_or_exit(args: argparse.Namespace) -> None:
    if args.yes:
        return
    print("[reset] This will archive, then clear maintainer-local donation test state:")
    print(f"[reset]   dataset candidate : {args.dataset_root}")
    print(f"[reset]   staging mirror    : {args.staging_dir}")
    print(f"[reset]   validation output : {args.validation_results}")
    print("[reset] It will NOT delete remote Hugging Face staging submissions.")
    answer = input("[reset] Type RESET to continue: ").strip()
    if answer != "RESET":
        raise SystemExit("[reset] aborted")


def main() -> int:
    p = argparse.ArgumentParser(description="Archive and clear local donation test state.")
    p.add_argument("--dataset-root", type=Path, default=Path("data_archive_release_v2"))
    p.add_argument("--staging-dir", type=Path, default=Path("hf_staging_download"))
    p.add_argument("--validation-results", type=Path, default=Path("results_v2_candidate"))
    p.add_argument("--backup-root", type=Path, default=Path(".donation_test_state_backups"))
    p.add_argument("--yes", action="store_true", help="skip the interactive RESET confirmation")
    args = p.parse_args()

    confirm_or_exit(args)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup = args.backup_root / stamp
    backup.mkdir(parents=True, exist_ok=False)

    archived = []
    for name, path in [
        ("data_archive_release_v2", args.dataset_root),
        ("hf_staging_download", args.staging_dir),
        ("results_v2_candidate", args.validation_results),
    ]:
        if copy_if_exists(path, backup / name):
            archived.append(path)

    for path in [args.dataset_root, args.staging_dir, args.validation_results]:
        remove_path(path)

    (args.dataset_root / "data" / "sessions").mkdir(parents=True, exist_ok=True)
    (args.dataset_root / "data" / "donations").mkdir(parents=True, exist_ok=True)

    print("[reset] archived local test state")
    print(f"[reset] backup : {backup.resolve()}")
    if archived:
        for path in archived:
            print(f"[reset] saved  : {path}")
    else:
        print("[reset] saved  : nothing found")
    print("[reset] cleared local maintainer artifacts")
    print(f"[reset] ready  : {args.dataset_root / 'data' / 'sessions'}")
    print("[reset] remote Hugging Face staging was not modified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
