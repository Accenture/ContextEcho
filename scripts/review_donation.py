"""One-command maintainer review for a staged ContextEcho donation.

Input is a downloaded Hugging Face staging folder:
  pending/submission-xxxx/

Default mode is technical-only and makes no API calls. Pass --run-quick to run
the 30-cell session validation gate.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SESSION_NAME = "session.redacted.jsonl"
MANIFEST_NAME = "manifest.json"
CONSENT_NAME = "CONSENT.md"


def safe_label(text: str) -> str:
    out = "".join(c if c.isalnum() or c in {"-", "_"} else "-" for c in text.strip())
    out = "-".join(part for part in out.split("-") if part)
    return out[:64] or "donor"


def default_label(manifest: dict, submission: Path) -> str:
    base = "anonymous-donor" if manifest.get("public_anonymous") else (manifest.get("credit_name") or manifest.get("contributor") or "donor")
    return safe_label(f"{base}-{submission.name}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_jsonl(path: Path) -> dict:
    lines = 0
    bad = []
    user = 0
    assistant = 0
    timestamps = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        lines += 1
        try:
            obj = json.loads(line)
        except Exception as e:
            if len(bad) < 10:
                bad.append({"line": line_no, "error": str(e)})
            continue
        t = obj.get("type")
        if t == "user":
            user += 1
        elif t == "assistant":
            assistant += 1
        ts = obj.get("timestamp")
        if isinstance(ts, str):
            timestamps.append(ts)
    return {
        "lines": lines,
        "invalid_lines": len(bad),
        "invalid_examples": bad,
        "user_rows": user,
        "assistant_rows": assistant,
        "first_timestamp": min(timestamps) if timestamps else "",
        "last_timestamp": max(timestamps) if timestamps else "",
    }


def run_cmd(cmd: list[str], env: dict | None = None, cwd: Path = REPO_ROOT) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout


def output_tail(text: str, limit: int = 4000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return "...[truncated]...\n" + text[-limit:]


def verify_session(path: Path, python: str) -> dict:
    rc, out = run_cmd([python, "-m", "donate.verify", str(path.resolve())], cwd=REPO_ROOT)
    return {"passed": rc == 0, "returncode": rc, "output": out}


def consent_ok(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    checks = {
        "right_to_donate": "I own this session" in text or "right to donate" in text,
        "no_confidential": "no client-confidential" in text and "NDA" in text,
        "reviewed_redaction": "redaction" in text.lower() and "reviewed" in text.lower(),
        "license": "CC-BY-SA-4.0" in text,
    }
    return {"passed": all(checks.values()), "checks": checks}


def validation_root(label: str, quick: bool, target: str) -> Path:
    kind = "session_validation_quick" if quick else "session_validation"
    return REPO_ROOT / "results_v2_candidate" / kind / label / target


def run_quick_validation(session: Path, label: str, python: str, env_file: Path | None) -> dict:
    cmd = [
        python,
        "experiments/e18_session_validation/run.py",
        "--session",
        str(session),
        "--label",
        label,
        "--quick",
    ]
    env = os.environ.copy()
    if env_file and env_file.exists():
        for raw in env_file.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    t0 = time.time()
    rc, out = run_cmd(cmd, env=env)
    return {"passed": rc == 0, "returncode": rc, "elapsed_sec": round(time.time() - t0, 1), "output": out}


def analyze_quick(label: str, python: str, target: str) -> dict:
    root = validation_root(label, quick=True, target=target)
    cmd = [
        python,
        "analysis/analyze_session_validation.py",
        "--root",
        str(root.relative_to(REPO_ROOT)),
        "--positions",
        "3",
        "--probes",
        "5",
        "--json",
    ]
    rc, out = run_cmd(cmd)
    try:
        data = json.loads(out)
    except Exception:
        data = {"raw_output": out}
    data["passed"] = rc == 0
    return data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Review one downloaded staging donation.")
    p.add_argument("submission", type=Path, help="pending/submission-* folder")
    p.add_argument("--label", default="", help="validation label; default is contributor plus submission id")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--run-quick", action="store_true", help="run 30-cell API validation gate")
    p.add_argument("--env-file", type=Path,
                   default=Path("/Users/xianzhong.ding/Library/CloudStorage/OneDrive-Accenture/Documents/mock_interview/me/projects/.env"))
    p.add_argument("--target", default="claude-sonnet-4-5")
    p.add_argument("--json", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sub = args.submission.expanduser()
    session = sub / SESSION_NAME
    manifest_path = sub / MANIFEST_NAME
    consent_path = sub / CONSENT_NAME

    report: dict = {
        "submission": str(sub),
        "files": {
            SESSION_NAME: session.exists(),
            MANIFEST_NAME: manifest_path.exists(),
            CONSENT_NAME: consent_path.exists(),
        },
        "checks": {},
        "metadata": {},
    }

    if manifest_path.exists():
        manifest = load_json(manifest_path)
        report["metadata"] = {
            "label": safe_label(args.label) if args.label else default_label(manifest, sub),
            "session_id": manifest.get("session_id"),
            "contributor": manifest.get("contributor"),
            "credit_name": manifest.get("credit_name"),
            "public_anonymous": bool(manifest.get("public_anonymous")),
            "institute": manifest.get("contributor_institute"),
            "agent": manifest.get("agent"),
            "model": manifest.get("model"),
            "org": manifest.get("org"),
            "domain": manifest.get("domain"),
            "language": manifest.get("language"),
            "records": manifest.get("records"),
            "turns": manifest.get("turns"),
            "compactions": manifest.get("compactions"),
            "privacy_tier": manifest.get("privacy_tier", "full_redacted"),
            "source_format": manifest.get("source_format"),
            "metadata_confidence": manifest.get("metadata_confidence", {}),
            "submitted_utc": manifest.get("submitted_utc"),
        }
    else:
        report["metadata"]["label"] = safe_label(args.label or sub.name)

    label = str(report["metadata"]["label"])
    report["checks"]["files_present"] = all(report["files"].values())

    if session.exists():
        jsonl = check_jsonl(session)
        report["checks"]["jsonl"] = jsonl
        report["checks"]["jsonl_valid"] = jsonl["invalid_lines"] == 0
        verify = verify_session(session, args.python)
        report["checks"]["verify"] = {
            "passed": verify["passed"],
            "returncode": verify["returncode"],
        }
    else:
        report["checks"]["jsonl_valid"] = False
        report["checks"]["verify"] = {"passed": False, "returncode": None}

    if consent_path.exists():
        report["checks"]["consent"] = consent_ok(consent_path)
    else:
        report["checks"]["consent"] = {"passed": False, "checks": {}}

    if manifest_path.exists() and session.exists():
        jsonl_lines = report["checks"].get("jsonl", {}).get("lines")
        records = report["metadata"].get("records")
        if records in {None, ""}:
            records = report["metadata"].get("turns")
        report["checks"]["manifest_matches_session"] = str(records) == str(jsonl_lines)
    else:
        report["checks"]["manifest_matches_session"] = False

    if args.run_quick:
        quick = run_quick_validation(session, label, args.python, args.env_file)
        report["checks"]["quick_validation_run"] = {
            "passed": quick["passed"],
            "returncode": quick["returncode"],
            "elapsed_sec": quick["elapsed_sec"],
            "output_tail": output_tail(quick["output"]),
        }
        report["checks"]["quick_validation"] = analyze_quick(label, args.python, args.target)

    technical_ok = (
        report["checks"].get("files_present")
        and report["checks"].get("jsonl_valid")
        and report["checks"].get("verify", {}).get("passed")
        and report["checks"].get("consent", {}).get("passed")
        and report["checks"].get("manifest_matches_session")
    )
    quick_ok = True
    if args.run_quick:
        quick_ok = bool(report["checks"].get("quick_validation", {}).get("acceptable"))
    report["decision"] = "ACCEPTABLE" if technical_ok and quick_ok else "CHECK_REQUIRED"

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("=== ContextEcho Donation Review ===")
        print(f"submission : {sub}")
        print(f"label      : {label}")
        meta = report["metadata"]
        for key in ("session_id", "contributor", "institute", "agent", "model", "org", "domain", "language", "turns", "records", "compactions", "privacy_tier", "source_format"):
            if meta.get(key) not in {None, ""}:
                print(f"{key:11s}: {meta[key]}")
        print("\nChecks:")
        print(f"  files present          : {'PASS' if report['checks']['files_present'] else 'FAIL'}")
        print(f"  jsonl valid            : {'PASS' if report['checks']['jsonl_valid'] else 'FAIL'}")
        if not report["checks"]["jsonl_valid"]:
            for item in report["checks"].get("jsonl", {}).get("invalid_examples", [])[:3]:
                print(f"    line {item['line']}: {item['error']}")
        print(f"  pii/secrets verify     : {'PASS' if report['checks']['verify']['passed'] else 'FAIL'}")
        print(f"  consent                : {'PASS' if report['checks']['consent']['passed'] else 'FAIL'}")
        print(f"  manifest/session match : {'PASS' if report['checks']['manifest_matches_session'] else 'FAIL'}")
        if args.run_quick:
            q = report["checks"].get("quick_validation", {})
            print(f"  quick validation       : {'PASS' if q.get('acceptable') else 'FAIL'}")
            print(f"  quick cells            : {q.get('scored_cells')}/{q.get('expected_cells')}")
            print(f"  quick gap              : {q.get('gap_filler_minus_claude'):+.3f}")
            print(f"  quick runtime sec      : {report['checks']['quick_validation_run']['elapsed_sec']}")
            if not q.get("acceptable"):
                tail = report["checks"]["quick_validation_run"].get("output_tail", "")
                if tail:
                    print("\nQuick validation output:")
                    print(tail)
        print(f"\nDecision: {report['decision']}")

    return 0 if report["decision"] == "ACCEPTABLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
