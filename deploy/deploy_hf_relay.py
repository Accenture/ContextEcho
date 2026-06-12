from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, get_token


SPACE_REPO_ID = os.environ.get(
    "CONTEXTECHO_RELAY_SPACE_REPO",
    "contextecho2026/context-echo-donation-relay",
)
STAGING_REPO = os.environ.get(
    "CONTEXTECHO_STAGING_REPO",
    "contextecho2026/persona-drift-staging",
)
ROOT = Path(__file__).resolve().parents[1]
SPACE_TEMPLATE_DIR = ROOT / "deploy" / "hf_relay_space"


def main() -> None:
    cached_token = get_token()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or cached_token
    staging_token = os.environ.get("HF_STAGING_TOKEN") or os.environ.get("CONTEXTECHO_DONATE_TOKEN") or cached_token
    if not token:
        raise SystemExit("Set HF_TOKEN to a Hugging Face token that can create/update the Space.")
    if not staging_token:
        raise SystemExit("Set HF_STAGING_TOKEN to a token that can write to the private staging dataset.")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,
    )
    api.add_space_secret(SPACE_REPO_ID, "HF_STAGING_TOKEN", staging_token)
    api.add_space_variable(SPACE_REPO_ID, "CONTEXTECHO_STAGING_REPO", STAGING_REPO)
    api.add_space_variable(SPACE_REPO_ID, "CONTEXTECHO_RELAY_STATE_DIR", "/data/.relay_state")
    with tempfile.TemporaryDirectory(prefix="contextecho-relay-space-") as td:
        bundle = Path(td)
        shutil.copytree(SPACE_TEMPLATE_DIR, bundle, dirs_exist_ok=True)
        shutil.copy2(ROOT / "requirements-relay.txt", bundle / "requirements-relay.txt")
        shutil.copytree(
            ROOT / "donate",
            bundle / "donate",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        api.upload_folder(
            repo_id=SPACE_REPO_ID,
            repo_type="space",
            folder_path=bundle,
            commit_message="Deploy ContextEcho donation relay",
            delete_patterns=["donate/**/__pycache__/**", "donate/**/*.pyc"],
        )
    api.restart_space(SPACE_REPO_ID)
    print(f"https://huggingface.co/spaces/{SPACE_REPO_ID}")
    print(f"https://{SPACE_REPO_ID.replace('/', '-')}.hf.space")


if __name__ == "__main__":
    main()
