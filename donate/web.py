"""Local browser wizard for ContextEcho donations.

Run:
    python -m donate.web

The server binds to 127.0.0.1 only. Raw sessions are read locally; only the
existing submit step can upload verified redacted artifacts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from donate import describe as describe_mod
from donate import discover as discover_mod
from donate import minimize as minimize_mod
from donate import redact as redact_mod
from donate import submit as submit_mod
from donate.adapters.base import is_redacted_artifact


DONATION_ROOT = Path.home() / "Downloads" / "ContextEcho_donations"
DONATION_REGISTRY = DONATION_ROOT / ".donated_sessions.json"


def safe_slug(text: str, default: str = "session") -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in text)
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return (cleaned[:64] or default).strip("-_") or default


def redacted_output_name(src: Path) -> str:
    stem = src.stem
    if stem.endswith(".redacted"):
        stem = stem[: -len(".redacted")]
    return f"{safe_slug(stem)}.redacted.jsonl"


def donation_output_dir(info: dict) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    agent = safe_slug(str(info.get("agent", "agent")).lower())
    project = safe_slug(str(info.get("project", "session")))
    return DONATION_ROOT / f"{stamp}-{agent}-{project}"


def session_key(path: str | Path) -> str:
    return hashlib.sha256(str(Path(path).expanduser()).encode("utf-8")).hexdigest()[:16]


def artifact_key(path: str | Path) -> str:
    p = Path(path).expanduser()
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_donation_registry() -> dict:
    try:
        data = json.loads(DONATION_REGISTRY.read_text())
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def load_donated_keys() -> set[str]:
    data = load_donation_registry()
    return {str(x) for x in data.get("source_keys", [])}


def load_donated_artifact_keys() -> set[str]:
    data = load_donation_registry()
    return {str(x) for x in data.get("artifact_keys", [])}


def save_donation_record(source_path: str | Path = "", artifact_path: str | Path = "", output: str = "") -> None:
    DONATION_ROOT.mkdir(parents=True, exist_ok=True)
    data = load_donation_registry()
    source_keys = {str(x) for x in data.get("source_keys", [])}
    artifact_keys = {str(x) for x in data.get("artifact_keys", [])}
    submissions = list(data.get("submissions", []))
    source = str(source_path or "")
    artifact = str(artifact_path or "")
    skey = session_key(source) if source else ""
    akey = artifact_key(artifact) if artifact and Path(artifact).expanduser().exists() else ""
    if skey:
        source_keys.add(skey)
    if akey:
        artifact_keys.add(akey)
    m = re.search(r"\[submit\] submission\s*:\s*(pending/submission-[^/\s]+/)", output)
    submissions.append({
        "submitted_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_key": skey,
        "artifact_key": akey,
        "submission": m.group(1) if m else "",
    })
    payload = {
        "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_keys": sorted(source_keys),
        "artifact_keys": sorted(artifact_keys),
        "submissions": submissions,
    }
    DONATION_REGISTRY.write_text(json.dumps(payload, indent=2) + "\n")


def save_donated_key(path: str | Path) -> None:
    save_donation_record(source_path=path)


def clear_donation_registry() -> bool:
    existed = DONATION_REGISTRY.exists()
    if existed:
        DONATION_REGISTRY.unlink()
    return existed


def already_submitted(source_path: str | Path = "", artifact_path: str | Path = "") -> bool:
    if source_path and session_key(source_path) in load_donated_keys():
        return True
    if artifact_path:
        p = Path(artifact_path).expanduser()
        if p.exists() and artifact_key(p) in load_donated_artifact_keys():
            return True
    return False


def parse_submit_output(output: str) -> dict:
    url = (re.search(r"https?://\S+", output) or [None])[0] or ""
    repo = (re.search(r"\[submit\] target repo\s*:\s*(.+)", output) or [None, ""])[1].strip()
    submission = (re.search(r"\[submit\] submission\s*:\s*(pending/submission-[^/\s]+/)", output) or [None, ""])[1].strip()
    uploads = [
        {"source": m.group(1).strip(), "target": m.group(2).strip()}
        for m in re.finditer(r"\[submit\]\s+(.+?)\s+->\s+(.+)", output)
    ]
    return {"url": url, "repo": repo, "submission": submission, "uploads": uploads}


def write_receipt(session: Path, source_path: str | Path, output: str) -> tuple[Path, dict]:
    stem = session.stem.replace(".redacted", "")
    manifest_path = session.with_name(f"{stem}.manifest.json")
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    parsed = parse_submit_output(output)
    receipt = {
        "submitted_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "submission": parsed["submission"],
        "review_url": parsed["url"],
        "target_repo": parsed["repo"],
        "source_path": str(source_path or ""),
        "redacted_file": str(session),
        "contributor": manifest.get("contributor", "anonymous"),
        "credit_name": manifest.get("credit_name", manifest.get("contributor", "anonymous")),
        "contributor_email": manifest.get("contributor_email", ""),
        "institute": manifest.get("contributor_institute", ""),
        "agent": manifest.get("agent", ""),
        "model": manifest.get("model", ""),
        "org": manifest.get("org", ""),
        "privacy_tier": manifest.get("privacy_tier", "full_redacted"),
        "turns": manifest.get("turns", ""),
        "compactions": manifest.get("compactions", ""),
        "uploads": parsed["uploads"],
    }
    lines = [
        "# ContextEcho Donation Receipt",
        "",
        "This receipt confirms that the local donation tool submitted verified redacted artifacts for maintainer review.",
        "",
        f"- Submitted UTC: {receipt['submitted_utc']}",
        f"- Submission: {receipt['submission'] or 'unknown'}",
        f"- Credit name: {receipt['credit_name']}",
        f"- Email: {receipt['contributor_email'] or 'not provided'}",
        f"- Institute: {receipt['institute'] or 'not provided'}",
        f"- Agent/model: {receipt['agent']} / {receipt['model']}",
        f"- Privacy tier: {receipt['privacy_tier']}",
        f"- Turns: {receipt['turns']}",
        f"- Compactions: {receipt['compactions']}",
        "",
        "Uploaded artifacts:",
    ]
    for item in receipt["uploads"]:
        lines.append(f"- {item['source']}")
    lines.extend([
        "",
        "Status: pending maintainer review. Credit is awarded after acceptance.",
        "",
    ])
    receipt_path = session.with_name("DONATION_RECEIPT.md")
    receipt_path.write_text("\n".join(lines), encoding="utf-8")
    return receipt_path, receipt


def annotate_donated(sessions: list[dict]) -> list[dict]:
    donated = load_donated_keys()
    out = []
    for session in sessions:
        row = dict(session)
        path = row.get("path")
        row["donated"] = bool(path and session_key(path) in donated)
        out.append(row)
    return out


def infer_domain(info: dict) -> str:
    project = str(info.get("project", "")).lower()
    if any(x in project for x in ("dashboard", "frontend", "web")):
        return "web-frontend"
    if any(x in project for x in ("paper", "research", "analysis")):
        return "research"
    if any(x in project for x in ("doc", "slide", "deck")):
        return "docs-writing"
    return "agentic-coding"


def infer_language(info: dict) -> str:
    project = str(info.get("project", "")).lower()
    if any(x in project for x in ("paper", "doc", "slide", "deck")):
        return "Markdown/docs"
    return "unknown"


def _fetch_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": "ContextEcho-donate-web/0.1"})
    with urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def _fetch_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "ContextEcho-donate-web/0.1"})
    with urlopen(req, timeout=5) as r:
        return r.read().decode("utf-8", errors="replace")


def _parse_donated_sessions(readme: str) -> int | None:
    patterns = [
        r"(\d[\d,]*)\s+redacted\s+donor\s+sessions?",
        r"(\d[\d,]*)\s+donated\s+sessions?",
        r"donated\s+sessions?\D{0,20}(\d[\d,]*)",
    ]
    for pattern in patterns:
        m = re.search(pattern, readme, re.IGNORECASE)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


def project_stats() -> dict:
    """Best-effort public project stats. Never block the donation flow."""
    stats = {
        "github_stars": None,
        "donated_sessions": None,
        "dataset_downloads": None,
        "dataset_likes": None,
    }
    try:
        gh = _fetch_json("https://api.github.com/repos/Accenture/ContextEcho")
        stats["github_stars"] = gh.get("stargazers_count")
    except Exception:
        pass
    try:
        readme = _fetch_text("https://raw.githubusercontent.com/Accenture/ContextEcho/main/README.md")
        stats["donated_sessions"] = _parse_donated_sessions(readme)
    except Exception:
        pass
    try:
        hf = _fetch_json("https://huggingface.co/api/datasets/contextecho2026/persona-drift-contextecho")
        stats["dataset_downloads"] = hf.get("downloads") or hf.get("downloadsAllTime")
        stats["dataset_likes"] = hf.get("likes")
    except Exception:
        pass
    return stats


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ContextEcho Donation Wizard</title>
  <style>
    :root { --ink:#111b18; --muted:#5f6662; --line:#dfe2da; --paper:#f6f7ed; --card:#fffef8; --accent:#17713f; --accent2:#e8a823; --soft:#eef6e8; --bad:#a63124; }
    body { margin:0; font:14px/1.35 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:radial-gradient(circle at 8% 0%,#fff7d5 0 18%,transparent 34%), radial-gradient(circle at 90% 8%,#dceedd 0 17%,transparent 34%), linear-gradient(135deg,#f7f3dd,#eef6ea 52%,#e5f1ee); }
    main { max-width:1480px; margin:0 auto; padding:14px 24px 28px; }
    h1 { margin:0 0 5px; font-size:clamp(28px,3.2vw,38px); letter-spacing:-.055em; line-height:1.02; }
    h2 { margin:0 0 6px; font-size:24px; letter-spacing:-.04em; line-height:1.08; }
    .hero, .card, .bottom-nav { background:rgba(255,255,250,.9); border:1px solid rgba(127,138,119,.28); border-radius:20px; box-shadow:0 14px 42px rgba(43,59,37,.11); backdrop-filter:blur(10px); }
    .hero { padding:24px 34px 22px; position:relative; overflow:hidden; }
    .hero-top { display:flex; justify-content:space-between; gap:20px; align-items:flex-start; }
    .card { padding:26px 34px; }
    .card.step { margin-top:16px; }
    .step { display:none; }
    .step.active { display:block; }
    .steps { display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:14px; margin-top:30px; align-items:center; }
    .step-pill { position:relative; display:flex; align-items:center; gap:11px; color:#6a6f6b; font-size:15px; font-weight:850; }
    .step-pill:after { content:""; height:3px; flex:1; border-radius:999px; background:#e2e4df; margin-left:6px; }
    .step-pill:last-child:after { display:none; }
    .step-num { display:grid; place-items:center; width:36px; height:36px; border-radius:999px; background:#e8e9e6; color:#555b58; font-weight:950; }
    .step-pill.active { color:var(--accent); }
    .step-pill.active .step-num, .step-pill.done .step-num { background:var(--accent); color:white; box-shadow:0 8px 20px rgba(23,113,63,.24); }
    .step-pill.active:after, .step-pill.done:after { background:var(--accent); }
    .hero-side { display:flex; align-items:flex-start; gap:16px; justify-content:flex-end; max-width:780px; }
    .privacy-note { color:var(--muted); font-size:13px; line-height:1.35; text-align:right; max-width:560px; padding-top:4px; white-space:nowrap; }
    .privacy-note strong { color:#13552f; }
    .hero-progress { display:flex; align-items:center; gap:14px; padding-top:4px; min-width:190px; justify-content:flex-end; }
    .progress-label { text-align:right; color:var(--muted); font-size:14px; }
    .progress-label strong { display:block; color:var(--accent); font-size:18px; }
    .ring { --pct:25; width:78px; height:78px; border-radius:50%; display:grid; place-items:center; background:conic-gradient(var(--accent) calc(var(--pct) * 1%), #e9ebe5 0); position:relative; font-weight:950; color:var(--accent); font-size:18px; }
    .ring:before { content:""; position:absolute; inset:7px; border-radius:50%; background:#fffef8; box-shadow:inset 0 0 0 1px rgba(0,0,0,.03); }
    .ring span { position:relative; }
    .muted { color:var(--muted); }
    .row { display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
    button { border:0; border-radius:10px; padding:12px 18px; background:var(--accent); color:white; font-weight:900; cursor:pointer; box-shadow:0 10px 20px rgba(23,113,63,.2); font-size:14px; }
    button.secondary { background:#e8eddc; color:var(--ink); box-shadow:none; }
    button:hover:not(:disabled) { transform:translateY(-1px); }
    button:disabled { opacity:.5; cursor:not-allowed; }
    input, textarea { width:100%; box-sizing:border-box; border:1px solid var(--line); border-radius:14px; padding:11px 13px; background:white; color:var(--ink); font:inherit; }
    input:focus, textarea:focus { outline:3px solid rgba(31,111,67,.16); border-color:#7cb67d; }
    label { display:block; font-weight:700; margin:12px 0 6px; }
    .pick-grid { display:grid; grid-template-columns:minmax(300px,.62fr) minmax(620px,1.38fr); gap:22px; margin-top:16px; }
    .pick-intro { min-height:342px; }
    .intro-head { display:flex; gap:22px; align-items:flex-start; padding-bottom:20px; border-bottom:1px solid var(--line); }
    .folder-icon { width:76px; height:76px; border-radius:18px; display:grid; place-items:center; background:linear-gradient(135deg,#eef6d4,#f7faeb); }
    .folder-icon:before { content:""; width:42px; height:29px; border:3px solid var(--accent); border-radius:6px; box-sizing:border-box; box-shadow:0 -10px 0 -6px var(--accent); }
    .stats { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin:22px 0 20px; }
    .stat-card { text-align:center; background:transparent; border:0; padding:0; min-height:0; }
    .stat-icon { width:52px; height:52px; margin:0 auto 8px; display:grid; place-items:center; border-radius:50%; background:#f6edd6; color:#d28b00; }
    .stat-icon svg { width:26px; height:26px; display:block; stroke:currentColor; fill:none; stroke-width:2.5; stroke-linecap:round; stroke-linejoin:round; }
    .stat-icon .icon-fill { fill:currentColor; stroke:none; }
    .stat-card:nth-child(2) .stat-icon { background:#e9f2e5; color:var(--accent); }
    .stat-card:nth-child(3) .stat-icon { background:#f6eadb; color:#dc4b30; }
    .stat-card:nth-child(4) .stat-icon { background:#efedf5; color:#7657a8; }
    .stat-value { font-size:23px; line-height:1; font-weight:950; letter-spacing:-.035em; }
    .stat-label { margin-top:5px; color:#3d4440; font-size:12px; font-weight:650; }
    .discover-main { width:100%; border-radius:10px; padding:14px 20px; font-size:18px; box-shadow:0 12px 24px rgba(23,113,63,.2); }
    .reset-donated { margin-top:12px; justify-content:center; }
    .reset-donated button { padding:8px 12px; font-size:12px; }
    .sessions-card { min-height:342px; }
    .session-head { display:flex; justify-content:space-between; align-items:center; gap:14px; margin-bottom:16px; }
    .session-head h2 { font-size:21px; }
    .count-badge { border-radius:10px; padding:6px 12px; color:var(--accent); background:#eaf4e5; font-weight:900; }
    .session-list { border:1px solid var(--line); border-radius:14px; overflow:hidden; background:white; }
    .session-table-head, .session-row { display:grid; grid-template-columns:40px minmax(250px,1fr) 110px 80px 66px 82px; gap:18px; align-items:center; }
    .session-table-head { padding:10px 14px; background:#f2f5ef; color:#5a625d; font-size:12px; font-weight:900; text-transform:uppercase; letter-spacing:.04em; border-bottom:1px solid var(--line); }
    .session-row { padding:12px 14px; border-bottom:1px solid var(--line); cursor:pointer; transition:.15s ease; }
    .session-row:last-child { border-bottom:0; }
    .session-row:hover, .session-row.selected { background:#f4f8ef; }
    .session-row.selected { box-shadow:inset 4px 0 0 var(--accent); }
    .session-row.donated-row { cursor:not-allowed; opacity:.72; background:#f7f9f4; }
    .session-row.donated-row:hover { background:#f7f9f4; }
    .session-icon { width:32px; height:32px; display:grid; place-items:center; border-radius:50%; background:#e8f1e4; color:var(--accent); font-weight:950; font-size:14px; }
    .session-title { font-weight:900; font-size:14px; }
    .session-date { color:#5f6662; font-size:13px; }
    .session-num { font-weight:900; font-size:15px; }
    .empty-sessions { padding:26px; text-align:center; color:var(--muted); }
    .bottom-nav { margin-top:16px; padding:12px 34px; display:flex; justify-content:space-between; align-items:center; gap:16px; }
    .tip { display:flex; gap:12px; align-items:center; color:#3f4843; }
    .tip:before { content:"?"; display:grid; place-items:center; width:22px; height:22px; border-radius:50%; border:2px solid var(--accent); color:var(--accent); font-weight:950; }
    .next-button { min-width:170px; font-size:16px; }
    .pill { display:inline-block; border-radius:999px; padding:3px 8px; font-size:12px; font-weight:800; background:#edf1e4; }
    .pill.best { background:#dff1d9; color:#13552f; }
    .pill.long { background:#e8ecd7; color:#5c5d16; }
    .pill.short { background:#f3e5d2; color:#7a420a; }
    .pill.donated { background:#dceafa; color:#1e4f87; }
    .inline-status { margin-top:10px; color:var(--muted); font-size:14px; }
    .result { display:none; border:1px solid var(--line); border-radius:18px; padding:16px; background:#fbfff4; margin-top:12px; }
    .result.show { display:block; }
    .success-panel { border:2px solid #1f6f43; background:linear-gradient(135deg,#e6f7df,#fffdf5); box-shadow:0 20px 70px rgba(31,111,67,.2); }
    .success-title { font-size:32px; font-weight:950; letter-spacing:-.04em; color:#13552f; }
    .success-subtitle { font-size:17px; color:#3e5d3d; margin-top:4px; }
    .result-head { display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; }
    .badge { display:inline-block; border-radius:999px; padding:5px 10px; font-weight:900; font-size:13px; }
    .badge.pass { background:#dff1d9; color:#13552f; }
    .badge.fail { background:#f6d8d3; color:#8a2118; }
    .field { margin-top:10px; }
    .field-label { font-size:12px; color:var(--muted); font-weight:800; text-transform:uppercase; letter-spacing:.04em; }
    .pathbox { margin-top:4px; padding:9px 10px; border-radius:10px; background:white; border:1px solid var(--line); font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; overflow:auto; }
    .metrics { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
    .metric { background:#edf3e8; border:1px solid #d5e4ce; border-radius:999px; padding:6px 10px; font-size:13px; }
    .selected-card { display:none; border:2px solid #7cb67d; background:#eef8e8; border-radius:18px; padding:14px; margin-top:14px; }
    .selected-card.show { display:block; }
    .selected-card-layout { display:flex; justify-content:space-between; gap:16px; align-items:flex-start; }
    .selected-card-main { min-width:0; flex:1; }
    .selected-card-action { flex:0 0 auto; }
    .search-panel { display:none; border:1px dashed #b8c9ad; border-radius:16px; padding:14px; margin-top:12px; background:#fffef7; }
    .search-panel.show { display:block; }
    .progress { width:100%; height:12px; border-radius:999px; overflow:hidden; background:#e5eadc; margin-top:12px; display:none; }
    .progress > div { height:100%; width:0%; background:linear-gradient(90deg,#1f6f43,#89b65b); transition:width .2s ease; }
    .danger { color:#7f241b; font-weight:800; background:#fff1ed; border:1px solid #f2c9c0; padding:10px 12px; border-radius:14px; }
    .ok { color:var(--accent); font-weight:800; }
    .hint { font-size:13px; color:var(--muted); margin-top:6px; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
    .topline { color:var(--muted); max-width:760px; font-size:18px; }
    .actions { justify-content:space-between; margin-top:18px; padding-top:16px; border-top:1px solid var(--line); }
    .compact-input-row label { margin:0; white-space:nowrap; }
    .compact-input-row input { flex:0 1 300px; min-width:220px; }
    .privacy-options { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:12px; }
    .privacy-card { border:1px solid var(--line); border-radius:16px; padding:12px; background:#fffef7; cursor:pointer; }
    .privacy-card:has(input:checked) { border-color:#1f6f43; background:#eef8e8; box-shadow:0 8px 22px rgba(31,111,67,.12); }
    .privacy-card input { width:auto; margin-right:7px; }
    @media (max-width:1000px) { .hero-top, .hero-side, .bottom-nav { align-items:flex-start; flex-direction:column; } .privacy-note { text-align:left; max-width:none; white-space:normal; } .hero-progress { justify-content:flex-start; } .pick-grid { grid-template-columns:1fr; } .session-table-head,.session-row { grid-template-columns:40px minmax(180px,1fr) 100px 74px 66px; } .session-fit { display:none; } }
    @media (max-width:700px) { main { padding:14px 10px 34px; } .hero,.card,.bottom-nav { border-radius:20px; padding:22px; } .grid { grid-template-columns:1fr; } .stats { grid-template-columns:repeat(2,minmax(0,1fr)); } .steps { grid-template-columns:1fr; gap:10px; margin-top:24px; } .step-pill:after { display:none; } .session-table-head,.session-row { grid-template-columns:36px 1fr 74px; } .session-date,.session-cmp,.session-fit { display:none; } .privacy-options { grid-template-columns:1fr; } .selected-card-layout { flex-direction:column; } .actions { justify-content:flex-start; } }
  </style>
</head>
<body>
<main>
  <section class="hero">
    <div class="hero-top">
      <div>
        <h1>ContextEcho Donation Wizard</h1>
        <div class="topline">Donate a coding-agent session in a few local-first steps.</div>
      </div>
      <div class="hero-side">
        <div class="privacy-note"><strong>Donor privacy:</strong> ContextEcho analyzes assistant behavior, not donor personality.<br>Default: <strong>full redacted</strong>. Stronger privacy: <strong>user-minimized</strong>.</div>
        <div class="hero-progress">
          <div class="progress-label"><strong id="stepLabel">Step 1 of 4</strong><span id="stepPercentText">25% complete</span></div>
          <div id="progressRing" class="ring" style="--pct:25"><span id="progressRingText">25%</span></div>
        </div>
      </div>
    </div>
    <div class="steps">
      <span id="pill1" class="step-pill active"><span class="step-num">1</span><span>Pick a Session</span></span>
      <span id="pill2" class="step-pill"><span class="step-num">2</span><span>Redact</span></span>
      <span id="pill3" class="step-pill"><span class="step-num">3</span><span>Describe</span></span>
      <span id="pill4" class="step-pill"><span class="step-num">4</span><span>Submit</span></span>
    </div>
  </section>

  <section id="step1" class="step active">
    <div class="pick-grid">
      <div class="card pick-intro">
        <div class="intro-head">
          <div class="folder-icon"></div>
          <div>
            <h2>1. Pick a Session</h2>
            <p class="muted">Choose a real session. Longer sessions with compactions provide the most benchmark value.</p>
          </div>
        </div>
        <div id="projectStats" class="stats" aria-live="polite">
          <div class="stat-card"><div class="stat-icon" data-icon="star"></div><div class="stat-value">...</div><div class="stat-label">GitHub Stars</div></div>
          <div class="stat-card"><div class="stat-icon" data-icon="download"></div><div class="stat-value">...</div><div class="stat-label">Dataset Downloads</div></div>
          <div class="stat-card"><div class="stat-icon" data-icon="heart"></div><div class="stat-value">...</div><div class="stat-label">Dataset Likes</div></div>
          <div class="stat-card"><div class="stat-icon" data-icon="gift"></div><div class="stat-value">...</div><div class="stat-label">Donated Sessions</div></div>
        </div>
        <button id="discoverBtn" class="discover-main">Discover Sessions</button>
        <div class="row reset-donated">
          <button id="clearDonatedBtn" class="secondary">Clear local donated labels</button>
        </div>
        <div id="discoverStatus" class="muted" style="margin-top:16px; text-align:center">Click discover to scan Claude/Codex sessions on this machine.</div>
        <div id="discoverProgress" class="progress"><div></div></div>
      </div>
      <div class="card sessions-card">
        <div class="session-head">
          <h2>Recently discovered sessions</h2>
          <span id="sessionCount" class="count-badge">0 found</span>
        </div>
        <div id="sessionList" class="session-list">
          <div class="session-table-head"><div>#</div><div>Name</div><div>Date</div><div>Turns</div><div>Cmp</div><div>Fit</div></div>
          <div class="empty-sessions">Click Discover Sessions to find local Claude/Codex sessions.</div>
        </div>
        <div id="pager" class="row" style="display:none; margin-top:18px; justify-content:center">
          <button id="prevPage" class="secondary">Previous</button>
          <span id="pageInfo" class="muted"></span>
          <button id="nextPage" class="secondary">Next</button>
        </div>
      </div>
    </div>
    <div class="bottom-nav">
      <div class="tip"><strong>Tip:</strong> Sessions with more turns and compactions are more valuable for the research community.</div>
      <button id="pickNext" class="next-button" disabled>Next: Redact  -&gt;</button>
    </div>
  </section>

  <section id="step2" class="card step">
    <h2>2. Redact + Verify</h2>
    <div id="selectedCard" class="selected-card"></div>
    <div class="danger">Only donate personal, internal tooling, or open-source sessions. Do not donate client-confidential/NDA data.</div>
    <p class="muted"><strong>ContextEcho analyzes assistant behavior, not donor personality.</strong> Choose how much of your own wording to keep.</p>
    <div class="privacy-options">
      <label class="privacy-card"><input type="radio" name="privacyTier" value="full_redacted" checked><strong>Full redacted</strong><div class="hint">Default. Keeps task flow after PII/secrets/custom terms are removed. Highest scientific fidelity.</div></label>
      <label class="privacy-card"><input type="radio" name="privacyTier" value="user_minimized"><strong>User-minimized</strong><div class="hint">Masks donor free-text after redaction. Assistant/tool behavior remains; lower detail, stronger privacy.</div></label>
    </div>
    <p class="muted">Automatic redaction covers common sensitive data such as paths, usernames, emails, names, phone numbers, IPs, URLs, API keys, tokens, and credential-like strings.</p>
    <label><input id="safeConfirm" type="checkbox" style="width:auto"> I confirm this session is safe to donate.</label>
    <div id="scrubRow" class="row compact-input-row" style="margin-top:12px">
      <label>Extra terms to scrub <span class="muted">(optional)</span></label>
      <input id="scrub" placeholder="your name, Project Codename" />
    </div>
    <div class="row" style="margin-top:12px">
      <button id="redactBtn" disabled>Redact and Verify</button>
    </div>
    <div id="redactProgress" class="progress"><div></div></div>
    <div id="redactResult" class="result"></div>
    <div id="searchPanel" class="search-panel">
      <label>Test search in redacted file <span class="muted">(optional)</span></label>
      <div class="row">
        <input id="searchTerms" placeholder="your name, Project Codename" style="flex:1; min-width:260px" />
        <button id="searchBtn" class="secondary">Search Redacted File</button>
      </div>
      <div id="searchProgress" class="progress"><div></div></div>
      <div id="searchResult" class="result"></div>
    </div>
    <div class="inline-status" id="redactStatus"></div>
    <label style="margin-top:14px"><input id="reviewConfirm" type="checkbox" style="width:auto" disabled> I reviewed the verify output and redacted file path; it is ready to describe.</label>
    <div class="row actions">
      <button id="redactPrev" class="secondary">Previous</button>
      <button id="redactNext" disabled>Next: Describe</button>
    </div>
  </section>

  <section id="step3" class="card step">
    <h2>3. Describe + Consent</h2>
    <p class="muted">Contributor info is used for credit, leaderboard accounting, and release acknowledgments. Leave blank to stay anonymous.</p>
    <p class="muted"><strong>Manifest</strong> records session metadata for the ledger. <strong>Consent</strong> records permission to donate the redacted session.</p>
    <div class="grid">
      <div><label>Name or GitHub/HF handle <span class="muted">(for credit, optional)</span></label><input id="contributorName" placeholder="anonymous" /></div>
      <div><label>Email <span class="muted">(optional)</span></label><input id="contributorEmail" placeholder="you@example.com" /></div>
    </div>
    <label>Institute <span class="muted">(optional)</span></label>
    <input id="contributorInstitute" placeholder="University / company / independent" />
    <div class="row" style="margin-top:12px">
      <button id="describeBtn" disabled>Write Manifest + Consent</button>
    </div>
    <div id="describeProgress" class="progress"><div></div></div>
    <div id="describeResult" class="result"></div>
    <div class="inline-status" id="describeStatus"></div>
    <div class="row actions">
      <button id="describePrev" class="secondary">Previous</button>
      <button id="describeNext" disabled>Next: Submit</button>
    </div>
  </section>

  <section id="step4" class="card step">
    <h2>4. Submit</h2>
    <p class="muted">Upload only the verified redacted session, manifest, and consent as a pull request for maintainer review.</p>
    <div class="row actions">
      <button id="submitPrev" class="secondary">Previous</button>
      <button id="submitBtn" disabled>Submit PR</button>
    </div>
    <div id="submitProgress" class="progress"><div></div></div>
    <div id="submitResult" class="result"></div>
    <div class="inline-status" id="submitStatus"></div>
  </section>
</main>
<script>
let sessions = [];
let selected = null;
let redacted = null;
let described = null;
let submitted = false;
let page = 0;
const pageSize = 5;
const $ = id => document.getElementById(id);
const donatedPaths = new Set(JSON.parse(localStorage.getItem('contextechoDonatedPaths') || '[]'));
let publicStats = {};
const statIcons = {
  star: '<svg viewBox="0 0 24 24" aria-hidden="true"><path class="icon-fill" d="M12 2.4l2.95 5.98 6.6.96-4.78 4.66 1.13 6.57L12 17.47l-5.9 3.1 1.13-6.57-4.78-4.66 6.6-.96L12 2.4z"/></svg>',
  download: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v11"/><path d="M7.5 9.5L12 14l4.5-4.5"/><path d="M5 17.5V20h14v-2.5"/></svg>',
  heart: '<svg viewBox="0 0 24 24" aria-hidden="true"><path class="icon-fill" d="M12 21s-7.25-4.45-9.35-8.7C.93 8.82 3.05 5 6.9 5c2.05 0 3.47 1.08 4.1 2.02C11.63 6.08 13.05 5 15.1 5c3.85 0 5.97 3.82 4.25 7.3C19.25 16.55 12 21 12 21z"/></svg>',
  gift: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 10h16v10H4z"/><path d="M3 6h18v4H3z"/><path d="M12 6v14"/><path d="M12 6c-2.8 0-4.7-.85-4.7-2.2C7.3 2.8 8.05 2 9.05 2 10.8 2 12 6 12 6z"/><path d="M12 6c2.8 0 4.7-.85 4.7-2.2 0-1-.75-1.8-1.75-1.8C13.2 2 12 6 12 6z"/></svg>'
};
function iconSvg(name){ return statIcons[name] || ''; }
function saveDonatedPaths(){ localStorage.setItem('contextechoDonatedPaths', JSON.stringify([...donatedPaths])); }
function privacyTier(){ return document.querySelector('input[name="privacyTier"]:checked')?.value || 'full_redacted'; }
function goStep(n){
  const pct = n * 25;
  for(let i=1;i<=4;i++){
    $('step'+i).classList.toggle('active', i===n);
    $('pill'+i).classList.toggle('active', i===n);
    $('pill'+i).classList.toggle('done', i<n);
  }
  $('stepLabel').textContent = `Step ${n} of 4`;
  $('stepPercentText').textContent = `${pct}% complete`;
  $('progressRing').style.setProperty('--pct', pct);
  $('progressRingText').textContent = `${pct}%`;
}
function refreshButtons(){
  const selectedDonated = !!(selected && (selected.donated || donatedPaths.has(selected.path)));
  $('pickNext').disabled = !selected;
  $('redactBtn').disabled = !(selected && $('safeConfirm').checked);
  $('reviewConfirm').disabled = !(redacted && redacted.verify_passed);
  $('redactNext').disabled = !(redacted && redacted.verify_passed && $('reviewConfirm').checked);
  $('describeBtn').disabled = !(redacted && redacted.verify_passed);
  $('describeNext').disabled = !described;
  $('submitBtn').disabled = !described || submitted || selectedDonated;
}
function fit(s){ const t=+s.turns||0,c=+s.compactions||0; return t>=1000&&c>0?'best':(t>=1000?'long':'short'); }
function turns(n){ n=+n||0; return n>=1000 ? (n/1000).toFixed(1)+'k' : String(n); }
function status(id, text){ $(id).textContent = text; }
function fmtStat(n){
  if(n === null || n === undefined || n === '') return '—';
  n = Number(n);
  if(!Number.isFinite(n)) return '—';
  if(n >= 1000000) return (n/1000000).toFixed(n >= 10000000 ? 0 : 1) + 'M';
  if(n >= 1000) return (n/1000).toFixed(n >= 10000 ? 0 : 1) + 'k';
  return String(n);
}
function escapeHtml(s){
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function renderProjectStats(){
  const cards = [
    ['star', 'GitHub Stars', publicStats.github_stars],
    ['download', 'Dataset Downloads', publicStats.dataset_downloads],
    ['heart', 'Dataset Likes', publicStats.dataset_likes],
    ['gift', 'Donated Sessions', publicStats.donated_sessions],
  ];
  $('projectStats').innerHTML = cards.map(([icon, label, value]) => `
    <div class="stat-card">
      <div class="stat-icon" data-icon="${escapeHtml(icon)}">${iconSvg(icon)}</div>
      <div class="stat-value">${escapeHtml(fmtStat(value))}</div>
      <div class="stat-label">${escapeHtml(label)}</div>
    </div>
  `).join('');
}
async function loadProjectStats(){
  renderProjectStats();
  try {
    const r = await fetch('/api/project_stats');
    if(!r.ok) return;
    publicStats = await r.json();
    renderProjectStats();
  } catch(e) {
    renderProjectStats();
  }
}
function renderRedactResult(data){
  const stats = data.stats || {};
  const entries = Object.entries(stats).sort((a,b)=>b[1]-a[1]);
  const metrics = entries.length
    ? entries.map(([k,v]) => `<span class="metric">${escapeHtml(k)}: <strong>${v}</strong></span>`).join('')
    : '<span class="metric">No detector matches</span>';
  $('redactResult').innerHTML = `
    <div class="result-head">
      <div><span class="badge ${data.verify_passed ? 'pass' : 'fail'}">${data.verify_passed ? 'Verified clean' : 'Verify failed'}</span></div>
      <div class="muted">${data.privacy_tier === 'user_minimized' ? 'Redaction + user minimization complete' : 'Redaction complete'}</div>
    </div>
    <div class="field"><div class="field-label">Redacted file</div><div class="pathbox">${escapeHtml(data.redacted_file)}</div></div>
    <div class="row" style="margin-top:8px"><button class="secondary" id="revealRedactedFile">Reveal File</button></div>
    <div class="field"><div class="field-label">Removed</div><div class="metrics">${metrics}</div></div>
  `;
  $('redactResult').classList.add('show');
  $('searchPanel').classList.add('show');
  $('searchResult').classList.remove('show');
  $('revealRedactedFile').onclick = () => post('/api/open_path', {path:data.redacted_file, reveal:true}).catch(e => status('redactStatus','ERROR: '+e.message));
}
function renderSelectedCard(s, idx){
  $('selectedCard').innerHTML = `
    <div class="selected-card-layout">
      <div class="selected-card-main">
        <div class="result-head">
          <div><strong>Selected #${idx + 1}: ${escapeHtml(s.project || 'unknown project')}</strong></div>
          <span class="pill ${fit(s)}">${fit(s)}</span>
        </div>
        <div class="metrics">
          <span class="metric">Agent: <strong>${escapeHtml(s.agent || '?')}</strong></span>
          <span class="metric">Model: <strong>${escapeHtml(s.model || '?')}</strong></span>
          <span class="metric">Turns: <strong>${turns(s.turns)}</strong></span>
          <span class="metric">Compactions: <strong>${s.compactions || 0}</strong></span>
          <span class="metric">Date: <strong>${escapeHtml(s.modified || '?')}</strong></span>
        </div>
      </div>
      <div class="selected-card-action"><button class="secondary" id="revealSourceFile">Reveal Source File</button></div>
    </div>
  `;
  $('selectedCard').classList.add('show');
  $('revealSourceFile').onclick = () => post('/api/open_path', {path:s.path, reveal:true}).catch(e => status('redactStatus','ERROR: '+e.message));
}
function renderSearchResult(data){
  const hits = data.results || [];
  const anyHit = hits.some(x => x.count > 0);
  const metrics = hits.length
    ? hits.map(x => `<span class="metric">${escapeHtml(x.term)}: <strong>${x.count}</strong></span>`).join('')
    : '<span class="metric">No terms entered</span>';
  $('searchResult').innerHTML = `
    <div class="result-head">
      <div><span class="badge ${anyHit ? 'fail' : 'pass'}">${anyHit ? 'Matches found' : 'No matches'}</span></div>
      <div class="muted">${anyHit ? 'Add matched terms to scrub and re-run redaction.' : 'Manual search passed for these terms.'}</div>
    </div>
    <div class="metrics">${metrics}</div>
  `;
  $('searchResult').classList.add('show');
}
function renderDescribeResult(data){
  $('describeResult').innerHTML = `
    <div class="result-head">
      <div><span class="badge pass">Files written</span></div>
      <div class="muted">Manifest and consent are ready.</div>
    </div>
    <div class="field"><div class="field-label">Manifest</div><div class="pathbox">${escapeHtml(data.manifest)}</div></div>
    <div class="field"><div class="field-label">Consent</div><div class="pathbox">${escapeHtml(data.consent)}</div></div>
  `;
  $('describeResult').classList.add('show');
}
function receiptEmailHref(receipt, receiptPath){
  const email = receipt.contributor_email || '';
  const publicId = (receipt.submission || '').replace(/^pending\//, '').replace(/\/$/, '') || 'unknown';
  const subject = `ContextEcho donation receipt ${publicId}`.trim();
  const body = [
    'ContextEcho donation receipt',
    '',
    `Submission ID: ${publicId}`,
    `Credit name: ${receipt.credit_name || 'anonymous'}`,
    `Agent/model: ${(receipt.agent || '')} / ${(receipt.model || '')}`,
    `Privacy tier: ${receipt.privacy_tier || 'full_redacted'}`,
    `Turns: ${receipt.turns || ''}`,
    `Compactions: ${receipt.compactions || ''}`,
    `Receipt file: ${receiptPath || ''}`,
    '',
    'Status: pending maintainer review. Credit is awarded after acceptance.'
  ].join('\n');
  return `mailto:${encodeURIComponent(email)}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
}
function renderSubmitResult(data){
  const receipt = data.receipt || {};
  const publicId = (receipt.submission || '').replace(/^pending\//, '').replace(/\/$/, '') || 'recorded locally';
  const uploads = (receipt.uploads || [])
    .map(m => `<span class="metric">${escapeHtml(m.source)}</span>`)
    .join('');
  const emailHref = receipt.contributor_email ? receiptEmailHref(receipt, data.receipt_path) : '';
  $('submitResult').innerHTML = `
    <div class="success-title">Submission received</div>
    <div class="success-subtitle">Your verified redacted session was uploaded for private maintainer review.</div>
    <div class="metrics">
      <span class="metric">Status: <strong>pending review</strong></span>
      <span class="metric">Credit: <strong>+2 pending</strong></span>
      <span class="metric">Novelty: <strong>+1 possible bonus</strong></span>
    </div>
    <div class="field"><div class="field-label">Submission ID</div><div class="pathbox">${escapeHtml(publicId)}</div><div class="hint">Save this ID for support. Maintainers can use it to find your private staging submission.</div></div>
    ${data.receipt_path ? `<div class="field"><div class="field-label">Receipt</div><div class="row"><button id="revealReceipt" class="secondary">Reveal Receipt</button>${emailHref ? `<a href="${escapeHtml(emailHref)}"><button class="secondary">Email Receipt</button></a>` : ''}</div><div class="pathbox">${escapeHtml(data.receipt_path)}</div><div class="hint">${emailHref ? 'Email opens your mail app with the receipt details; no email is sent by the local tool.' : 'No email was provided, so the receipt was saved locally only.'}</div></div>` : ''}
    ${uploads ? `<div class="field"><div class="field-label">Submitted files</div><div class="metrics">${uploads}</div></div>` : ''}
    <div class="row" style="margin-top:14px"><button id="submitAnother" class="secondary">Submit another session</button></div>
  `;
  $('submitResult').classList.add('show', 'success-panel');
  if(data.receipt_path) $('revealReceipt').onclick = () => post('/api/open_path', {path:data.receipt_path, reveal:true}).catch(e => status('submitStatus','ERROR: '+e.message));
  $('submitAnother').onclick = () => goStep(1);
}
function setProgress(pct){
  $('discoverProgress').style.display = 'block';
  $('discoverProgress').firstElementChild.style.width = Math.max(0, Math.min(100, pct)) + '%';
}
function setBusy(id, on, pct=35){
  const el = $(id);
  el.style.display = on ? 'block' : 'none';
  el.firstElementChild.style.width = on ? pct + '%' : '0%';
}
async function post(url, body){
  const r = await fetch(url, {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(body)});
  const data = await r.json();
  if(!r.ok) throw new Error(data.error || r.statusText);
  return data;
}
function renderSessions(){
  const list = $('sessionList');
  list.innerHTML = '';
  const start = page * pageSize;
  const rows = sessions.slice(start, start + pageSize);
  $('sessionCount').textContent = `${sessions.length} found`;
  if(!rows.length){
    list.innerHTML = '<div class="session-table-head"><div>#</div><div>Name</div><div>Date</div><div>Turns</div><div>Cmp</div><div>Fit</div></div><div class="empty-sessions">No sessions found yet. Click Discover Sessions to scan this machine.</div>';
  }
  if(rows.length){
    list.innerHTML = '<div class="session-table-head"><div>#</div><div>Name</div><div>Date</div><div>Turns</div><div>Cmp</div><div>Fit</div></div>';
  }
  rows.forEach((s,i) => {
    const idx = start + i;
    const row = document.createElement('div');
    const donated = !!s.donated || donatedPaths.has(s.path);
    row.className = donated ? 'session-row donated-row' : 'session-row';
    row.innerHTML = `
      <div class="session-icon">${idx + 1}</div>
      <div>
        <div class="session-title">${escapeHtml(s.agent || 'Session')} - ${escapeHtml(s.project || 'unknown project')} ${donated ? '<span class="pill donated">donated</span>' : ''}</div>
      </div>
      <div class="session-date">${escapeHtml(s.modified || '?')}</div>
      <div class="session-turns"><div class="session-num">${turns(s.turns)}</div></div>
      <div class="session-cmp"><div class="session-num">${s.compactions || 0}</div></div>
      <div class="session-fit"><span class="pill ${fit(s)}">${fit(s)}</span></div>
    `;
    if (selected && selected.path === s.path && !donated) row.classList.add('selected');
    row.onclick = () => {
      if(donated){
        status('discoverStatus', 'This session is already marked donated locally. Use Clear local donated labels only if the previous submission failed.');
        return;
      }
      document.querySelectorAll('.session-row.selected').forEach(x=>x.classList.remove('selected'));
      row.classList.add('selected'); selected = s;
      redacted = null; described = null; submitted = !!donated;
      renderSelectedCard(s, idx);
      status('redactStatus', donated ? 'This session is already marked donated locally. Pick a different session to avoid duplicate submissions.' : '');
      status('discoverStatus', '');
      refreshButtons();
    };
    list.appendChild(row);
  });
  const totalPages = Math.max(1, Math.ceil(sessions.length / pageSize));
  $('pageInfo').textContent = `Page ${page + 1} of ${totalPages} · showing ${sessions.length ? start + 1 : 0}-${Math.min(start + pageSize, sessions.length)} of ${sessions.length}`;
  $('prevPage').disabled = page <= 0;
  $('nextPage').disabled = page >= totalPages - 1;
}
$('discoverBtn').onclick = async () => {
  $('discoverBtn').disabled = true;
  status('discoverStatus','Scanning local session logs. This can take a minute for large histories...');
  setProgress(2);
  try {
    const max = '50';
    const r = await fetch('/api/discover_stream?max_per_agent=' + max);
    if(!r.ok) throw new Error(await r.text());
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let final = null;
    while(true){
      const {done, value} = await reader.read();
      if(done) break;
      buffer += decoder.decode(value, {stream:true});
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for(const line of lines){
        if(!line.trim()) continue;
        const ev = JSON.parse(line);
        if(ev.event === 'adapter_start'){
          status('discoverStatus', `Scanning ${ev.agent}...`);
        } else if(ev.event === 'inspect'){
          const limit = ev.adapter_limit || 50;
          const pct = max === 'all' ? Math.min(90, 5 + ev.inspected) : Math.min(90, ((ev.inspected || 0) / (limit * 2)) * 90);
          setProgress(pct);
          status('discoverStatus', `${ev.agent}: inspected ${ev.adapter_inspected}${ev.adapter_limit ? '/' + ev.adapter_limit : ''}; found ${ev.found} usable so far.`);
        } else if(ev.event === 'adapter_done'){
          status('discoverStatus', `${ev.agent}: done. ${ev.found} usable sessions so far.`);
        } else if(ev.event === 'done'){
          final = ev;
          setProgress(100);
        }
      }
    }
    sessions = (final && final.sessions) || [];
    page = 0;
    status('discoverStatus', `Found ${sessions.length} usable sessions. Click a row to select.`);
    renderSessions();
    $('pager').style.display = sessions.length > pageSize ? 'flex' : 'none';
  } catch(e) { status('discoverStatus','ERROR: '+e.message); }
  finally { $('discoverBtn').disabled = false; }
};
$('clearDonatedBtn').onclick = async () => {
  const ok = confirm('Clear local donated labels on this browser and machine? This does not delete or retract submitted data. It may allow resubmission; maintainers may reject duplicates.');
  if(!ok) return;
  try {
    await post('/api/clear_donated_labels', {});
    donatedPaths.clear();
    saveDonatedPaths();
    sessions = sessions.map(s => ({...s, donated:false}));
    if(selected) selected.donated = false;
    submitted = false;
    renderSessions();
    refreshButtons();
    status('discoverStatus', 'Local donated labels cleared. Submitted data and maintainer records are unchanged.');
  } catch(e) {
    status('discoverStatus','ERROR: '+e.message);
  }
};
$('prevPage').onclick = () => { if(page > 0){ page--; renderSessions(); } };
$('nextPage').onclick = () => { if((page + 1) * pageSize < sessions.length){ page++; renderSessions(); } };
$('safeConfirm').onchange = refreshButtons;
$('reviewConfirm').onchange = refreshButtons;
document.querySelectorAll('input[name="privacyTier"]').forEach(el => {
  el.onchange = () => {
    if(redacted){
      redacted = null;
      described = null;
      $('reviewConfirm').checked = false;
      $('redactResult').classList.remove('show');
      $('searchPanel').classList.remove('show');
      $('searchResult').classList.remove('show');
      status('redactStatus', 'Privacy mode changed. Click Redact and Verify again before moving on.');
    }
    refreshButtons();
  };
});
$('scrub').oninput = () => {
  if(redacted){
    $('reviewConfirm').checked = false;
    $('redactResult').classList.remove('show');
    $('searchPanel').classList.remove('show');
    $('searchResult').classList.remove('show');
    status('redactStatus', 'Scrub terms changed. Click Redact and Verify again before moving on.');
  }
  refreshButtons();
};
$('pickNext').onclick = () => goStep(2);
$('redactPrev').onclick = () => goStep(1);
$('redactNext').onclick = () => goStep(3);
$('describePrev').onclick = () => goStep(2);
$('describeNext').onclick = () => goStep(4);
$('submitPrev').onclick = () => goStep(3);
$('searchBtn').onclick = async () => {
  if(!redacted) return;
  setBusy('searchProgress', true, 55);
  try {
    const data = await post('/api/search_redacted', {redacted_file:redacted.redacted_file, terms:$('searchTerms').value});
    renderSearchResult(data);
  } catch(e) { status('redactStatus','ERROR: '+e.message); }
  finally { setBusy('searchProgress', false); }
};
$('redactBtn').onclick = async () => {
  if(!selected) return;
  $('redactBtn').disabled = true;
  $('redactResult').classList.remove('show');
  $('searchPanel').classList.remove('show');
  $('searchResult').classList.remove('show');
  setBusy('redactProgress', true, 30);
  status('redactStatus','Redacting locally, then running verify. This may take several minutes...');
  try {
    redacted = await post('/api/redact', {path:selected.path, scrub:$('scrub').value, auto:selected, confirm_safe:$('safeConfirm').checked, privacy_tier:privacyTier()});
    setBusy('redactProgress', true, 100);
    described = null;
    submitted = false;
    $('reviewConfirm').checked = false;
    renderRedactResult(redacted);
    status('redactStatus', redacted.verify_passed ? 'Review the result above. Add more scrub terms if needed, then check the review box to continue.' : 'Verify failed. Add more scrub terms and re-run.');
    refreshButtons();
  } catch(e) { status('redactStatus','ERROR: '+e.message); }
  finally { setBusy('redactProgress', false); refreshButtons(); }
};
$('describeBtn').onclick = async () => {
  if(!redacted) return;
  $('describeBtn').disabled = true;
  $('describeResult').classList.remove('show');
  setBusy('describeProgress', true, 60);
  try {
    described = await post('/api/describe', {
      redacted_file:redacted.redacted_file,
      auto:selected,
      privacy_tier:redacted.privacy_tier || privacyTier(),
      contributor:$('contributorName').value,
      email:$('contributorEmail').value,
      institute:$('contributorInstitute').value
    });
    renderDescribeResult(described);
    status('describeStatus', 'Review the generated files, then continue to submit.');
    status('submitStatus', submitted ? 'This session is already marked donated locally.' : 'Ready to submit.');
    refreshButtons();
  } catch(e) { status('describeStatus','ERROR: '+e.message); }
  finally { setBusy('describeProgress', false); refreshButtons(); }
};
$('submitBtn').onclick = async () => {
  if(!redacted || !confirm('Upload verified redacted artifacts as a PR?')) return;
  $('submitBtn').disabled = true;
  $('submitResult').classList.remove('show');
  setBusy('submitProgress', true, 45);
  status('submitStatus','Submitting PR...');
  try {
    const data = await post('/api/submit', {redacted_file:redacted.redacted_file, source_path:selected ? selected.path : ''});
    submitted = true;
    if(selected && selected.path){ selected.donated = true; donatedPaths.add(selected.path); saveDonatedPaths(); renderSessions(); }
    renderSubmitResult(data);
    status('submitStatus', 'Submission marked donated locally. Pick another session to submit more.');
    refreshButtons();
  }
  catch(e) { status('submitStatus','ERROR: '+e.message); }
  finally { setBusy('submitProgress', false); refreshButtons(); }
};
loadProjectStats();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "ContextEchoDonateWeb/0.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[web] " + (fmt % args) + "\n")

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        n = int(self.headers.get("content-length", "0"))
        return json.loads(self.rfile.read(n).decode() or "{}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = INDEX_HTML.encode()
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/discover":
            qs = parse_qs(parsed.query)
            raw_max = qs.get("max_per_agent", ["50"])[0]
            max_per_agent = None if raw_max == "all" else int(raw_max)
            sessions = discover_mod.discover(max_per_agent=max_per_agent, progress=False)
            self._json({"sessions": annotate_donated(sessions)})
            return
        if parsed.path == "/api/project_stats":
            self._json(project_stats())
            return
        if parsed.path == "/api/discover_stream":
            qs = parse_qs(parsed.query)
            raw_max = qs.get("max_per_agent", ["50"])[0]
            max_per_agent = None if raw_max == "all" else int(raw_max)
            self.send_response(200)
            self.send_header("content-type", "application/x-ndjson")
            self.send_header("cache-control", "no-store")
            self.end_headers()
            for event in discover_mod.discover_iter(max_per_agent=max_per_agent):
                if event.get("event") == "done":
                    event = dict(event)
                    event["sessions"] = annotate_donated(list(event.get("sessions") or []))
                self.wfile.write((json.dumps(event) + "\n").encode())
                self.wfile.flush()
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        try:
            if self.path == "/api/redact":
                self._handle_redact()
            elif self.path == "/api/describe":
                self._handle_describe()
            elif self.path == "/api/submit":
                self._handle_submit()
            elif self.path == "/api/open_path":
                self._handle_open_path()
            elif self.path == "/api/search_redacted":
                self._handle_search_redacted()
            elif self.path == "/api/clear_donated_labels":
                self._handle_clear_donated_labels()
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": str(e)}, 400)

    def _handle_redact(self) -> None:
        data = self._read_json()
        if not data.get("confirm_safe"):
            raise ValueError("safety confirmation is required")
        src = Path(data.get("path", "")).expanduser()
        if not src.exists():
            raise ValueError(f"not found: {src}")
        if is_redacted_artifact(src):
            raise ValueError("selected file already looks redacted; choose the original session log")
        auto = data.get("auto") or discover_mod.inspect_session(src)
        scrub_terms = {t.strip() for t in str(data.get("scrub", "")).split(",") if t.strip()}
        privacy_tier = str(data.get("privacy_tier") or "full_redacted")
        if privacy_tier not in {"full_redacted", "user_minimized"}:
            raise ValueError("invalid privacy tier")
        out_dir = donation_output_dir(auto)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / redacted_output_name(src)
        stats = redact_mod.redact_file(src, out, scrub_terms, progress=False)
        if privacy_tier == "user_minimized":
            min_stats = minimize_mod.minimize_file(out, out)
            stats.update({f"minimize_{k}": v for k, v in min_stats.items()})
        verify_ok = submit_mod.verify_passed(out)
        self._json({
            "redacted_file": str(out),
            "output_dir": str(out_dir),
            "stats": stats,
            "privacy_tier": privacy_tier,
            "verify_passed": verify_ok,
        })

    def _handle_describe(self) -> None:
        data = self._read_json()
        session = Path(data.get("redacted_file", "")).expanduser()
        if not session.exists():
            raise ValueError(f"not found: {session}")
        manifest, consent, _ = describe_mod.write_manifest_and_consent(
            session=session,
            auto=data.get("auto") or {},
            domain=infer_domain(data.get("auto") or {}),
            language=infer_language(data.get("auto") or {}),
            contributor=str(data.get("contributor", "") or "anonymous"),
            email=str(data.get("email", "") or ""),
            institute=str(data.get("institute", "") or ""),
            privacy_tier=str(data.get("privacy_tier") or "full_redacted"),
        )
        self._json({"manifest": str(manifest), "consent": str(consent)})

    def _handle_submit(self) -> None:
        data = self._read_json()
        session = Path(data.get("redacted_file", "")).expanduser()
        if not session.exists():
            raise ValueError(f"not found: {session}")
        source_path = data.get("source_path")
        if already_submitted(source_path, session):
            self._json({
                "error": (
                    "This session or redacted artifact is already marked submitted locally. "
                    "Pick another session, or use Clear local donated labels only if the previous "
                    "submission truly failed."
                )
            }, 409)
            return
        # Capture submit's terminal-style output for the browser.
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = submit_mod.main([str(session)])
        output = buf.getvalue()
        if rc != 0:
            self._json({"error": output or f"submit failed with code {rc}"}, 400)
            return
        save_donation_record(source_path=source_path or "", artifact_path=session, output=output)
        receipt_path, receipt = write_receipt(session, source_path or "", output)
        self._json({"output": output, "receipt_path": str(receipt_path), "receipt": receipt})

    def _handle_clear_donated_labels(self) -> None:
        existed = clear_donation_registry()
        self._json({
            "cleared": True,
            "server_registry_existed": existed,
            "message": "Local donated labels cleared. Submitted data and maintainer records are unchanged.",
        })

    def _handle_open_path(self) -> None:
        data = self._read_json()
        path = Path(data.get("path", "")).expanduser()
        if not path.exists():
            raise ValueError(f"not found: {path}")
        if sys.platform == "darwin":
            cmd = ["open", "-R", str(path)] if data.get("reveal") else ["open", str(path)]
        elif sys.platform.startswith("linux"):
            target = path.parent if data.get("reveal") and path.is_file() else path
            cmd = ["xdg-open", str(target)]
        elif sys.platform.startswith("win"):
            cmd = ["explorer", f"/select,{path}"] if data.get("reveal") else ["explorer", str(path)]
        else:
            raise ValueError(f"open path is unsupported on {sys.platform}")
        subprocess.run(cmd, check=False)
        self._json({"ok": True})

    def _handle_search_redacted(self) -> None:
        data = self._read_json()
        path = Path(data.get("redacted_file", "")).expanduser()
        if not path.exists():
            raise ValueError(f"not found: {path}")
        if not is_redacted_artifact(path):
            raise ValueError("search is only allowed on redacted donation artifacts")
        terms = [t.strip() for t in str(data.get("terms", "")).split(",") if t.strip()]
        text = path.read_text(encoding="utf-8", errors="replace")
        self._json({"results": [{"term": term, "count": text.count(term)} for term in terms]})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the local ContextEcho donation web wizard.")
    p.add_argument("--host", default="127.0.0.1", help="bind host; default is local-only")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--no-open", action="store_true", help="do not open a browser")
    args = p.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[web] ContextEcho donation wizard: {url}")
    print("[web] Raw sessions stay local. Press Ctrl-C to stop.")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
