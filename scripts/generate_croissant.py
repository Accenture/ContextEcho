"""Emit Croissant 1.0 metadata for the ContextEcho release tree.

Croissant is JSON-LD; we don't need the `mlcroissant` package — we just
produce the JSON-LD directly per the spec:
    https://docs.mlcommons.org/croissant/docs/croissant-spec.html

The output describes:
  - 3 donor session JSONLs as one FileSet (sessions)
  - The per-cell JSON tree as a second FileSet (cells)
  - One RecordSet per FileSet documenting the field schema

After writing, validate at:
    https://huggingface.co/spaces/MLCommons/croissant-validator
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RELEASE_ROOT = PROJECT_ROOT / "data_archive_release"
OUT_PATH = RELEASE_ROOT / "croissant.json"

# The Croissant @id of each FileObject inside the dataset.
SESSION_FILES = [
    ("session_raw_transcript.jsonl",
     "Headline 9,716-turn donated Claude Code session (Donor 1)"),
    ("session_chainassemble.jsonl",
     "ChainAssemble 3,746-turn replication session (Donor 2)"),
    ("session_proeng.jsonl",
     "ProEng 4,918-turn replication session (Donor 3)"),
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_croissant() -> dict:
    sessions_dir = RELEASE_ROOT / "data" / "sessions"
    results_dir = RELEASE_ROOT / "results"

    # Per-session FileObject entries — one per donor JSONL.
    session_files = []
    for name, desc in SESSION_FILES:
        path = sessions_dir / name
        size = path.stat().st_size
        digest = sha256_of(path)
        session_files.append({
            "@type": "cr:FileObject",
            "@id": f"sessions/{name}",
            "name": name,
            "description": desc,
            "contentUrl": f"data/sessions/{name}",
            "encodingFormat": "application/jsonlines",
            "contentSize": str(size),
            "sha256": digest,
        })

    # FileSet wrapping all 3 session JSONLs so the session-turns RecordSet
    # below can point at a single source.
    sessions_fileset = {
        "@type": "cr:FileSet",
        "@id": "sessions",
        "name": "donor-sessions",
        "description": (
            f"Three anonymized donated Claude Code sessions, "
            f"{len(session_files)} JSONL files total."
        ),
        "encodingFormat": "application/jsonlines",
        "includes": "data/sessions/*.jsonl",
    }

    # The per-cell tree is too large to enumerate file-by-file; we
    # describe it as a single FileSet pointing to a glob over the
    # results directory. Reviewers extract per-file metadata from the
    # tree's MANIFEST.json.
    cell_count = sum(1 for _ in results_dir.rglob("*.json"))
    cells_fileset = {
        "@type": "cr:FileSet",
        "@id": "cells",
        "name": "per-cell-evaluations",
        "description": (
            f"{cell_count:,} per-cell JSON evaluation files organized by "
            "experiment / target / position / paraphrase / arm. See "
            "results/MANIFEST.json for a flat index with provenance."
        ),
        "encodingFormat": "application/json",
        "includes": "results/**/*.json",
    }

    # RecordSets — describe the schema readable inside each FileSet.
    # Each Field declares `source` so Croissant validators know where in the
    # underlying JSON/JSONL to extract that field.

    def _session_field(field_id, name, data_type, json_path, description):
        return {
            "@type": "cr:Field",
            "@id": f"session-turns/{field_id}",
            "name": name,
            "dataType": data_type,
            "description": description,
            "source": {
                "fileSet": {"@id": "sessions"},
                "extract": {"jsonPath": json_path},
            },
        }

    def _cell_field(field_id, name, data_type, json_path, description):
        return {
            "@type": "cr:Field",
            "@id": f"evaluation-cells/{field_id}",
            "name": name,
            "dataType": data_type,
            "description": description,
            "source": {
                "fileSet": {"@id": "cells"},
                "extract": {"jsonPath": json_path},
            },
        }

    sessions_recordset = {
        "@type": "cr:RecordSet",
        "@id": "session-turns",
        "name": "session-turns",
        "description": (
            "One record per turn in a donor session. Each line of each "
            "JSONL file is one turn."
        ),
        "field": [
            _session_field("role", "role", "sc:Text", "$.role",
                           "Speaker role: 'user' or 'assistant' or 'system'."),
            _session_field("content", "content", "sc:Text", "$.content",
                           "Verbatim turn content (post-PII-redaction)."),
            _session_field("sessionId", "sessionId", "sc:Text", "$.sessionId",
                           "Redacted session UUID placeholder (`<SESSION_UUID>`)."),
            _session_field("turn_idx", "turn_idx", "sc:Integer", "$.turn_idx",
                           "Zero-indexed turn order within the session."),
        ],
    }

    cells_recordset = {
        "@type": "cr:RecordSet",
        "@id": "evaluation-cells",
        "name": "evaluation-cells",
        "description": (
            "One record per (target × position × arm × paraphrase) "
            "evaluation cell."
        ),
        "field": [
            _cell_field("cell_id", "cell_id", "sc:Text", "$.cell_id",
                        "target/position/arm/paraphrase[/stressor]."),
            _cell_field("target_model_id", "target_model_id", "sc:Text", "$.target_model_id",
                        "Provider-namespaced model id (e.g., anthropic/claude-sonnet-4-6)."),
            _cell_field("position", "position", "sc:Text", "$.position",
                        "Measurement position label (P0_start … P5_pre_C6 for cross-compaction)."),
            _cell_field("arm", "arm", "sc:Text", "$.arm",
                        "Experimental arm: claude_session | filler | anchor_strong | etc."),
            _cell_field("paraphrase_idx", "paraphrase_idx", "sc:Integer", "$.paraphrase_idx",
                        "Paraphrase index 0..n within a cell."),
            _cell_field("prompt_text", "prompt_text", "sc:Text", "$.prompt_text",
                        "Verbatim probe text sent to the model."),
            _cell_field("response_text", "response_text", "sc:Text", "$.response_text",
                        "Verbatim model response."),
            _cell_field("judge_score", "judge_score", "sc:Integer", "$.judge_score",
                        "0..3 on the 4-point assistant-register rubric (where applicable)."),
            _cell_field("compliance_pass", "compliance_pass", "sc:Boolean", "$.compliance_pass",
                        "is_no_preamble regex pass (where applicable)."),
            _cell_field("len_chars", "len_chars", "sc:Integer", "$.len_chars",
                        "Raw character length of response_text."),
        ],
    }

    return {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "citeAs": "cr:citeAs",
            "column": "cr:column",
            "conformsTo": "dct:conformsTo",
            "cr": "http://mlcommons.org/croissant/",
            "data": {"@id": "cr:data", "@type": "@json"},
            "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
            "dct": "http://purl.org/dc/terms/",
            "examples": {"@id": "cr:examples", "@type": "@json"},
            "extract": "cr:extract",
            "field": "cr:field",
            "fileProperty": "cr:fileProperty",
            "fileObject": "cr:fileObject",
            "fileSet": "cr:fileSet",
            "format": "cr:format",
            "includes": "cr:includes",
            "isLiveDataset": "cr:isLiveDataset",
            "jsonPath": "cr:jsonPath",
            "key": "cr:key",
            "md5": "cr:md5",
            "parentField": "cr:parentField",
            "path": "cr:path",
            "recordSet": "cr:recordSet",
            "references": "cr:references",
            "regex": "cr:regex",
            "repeated": "cr:repeated",
            "replace": "cr:replace",
            "sc": "https://schema.org/",
            "separator": "cr:separator",
            "source": "cr:source",
            "subField": "cr:subField",
            "transform": "cr:transform",
            "rai": "http://mlcommons.org/croissant/RAI/",
            "prov": "http://www.w3.org/ns/prov#",
        },
        "@type": "sc:Dataset",
        "name": "ContextEcho",
        "description": (
            "ContextEcho is the per-cell evaluation corpus and donated "
            "session-prefix set for measuring persona drift in long "
            "agentic-coding sessions across 23 frontier LLM targets from "
            "10 organizations. Includes 3 redacted real Claude Code "
            "sessions (3,746–9,716 turns) and ~41,921 per-cell JSON "
            "evaluations spanning probe-surface (judge-scored), "
            "stressor-surface (judge-free regex compliance + length "
            "ratio), A-anchor mitigation, cross-judge audit, drift-onset "
            "sweep, SWE-Bench-style continuation, and TerminalBench "
            "fresh-task null."
        ),
        "url": "https://anonymous.4open.science/r/persona_drift_neurips-E541/",
        "license": "https://creativecommons.org/licenses/by-sa/4.0/",
        "version": "1.0.0",
        "citeAs": (
            "@inproceedings{contextecho2026, "
            "title={ContextEcho: A Benchmark for Persona Drift in Long "
            "Agentic-Coding Sessions}, author={Anonymous}, "
            "booktitle={NeurIPS 2026 Datasets and Benchmarks Track "
            "(under review)}, year={2026}}"
        ),
        "datePublished": "2026-05-05",
        "creator": {
            "@type": "sc:Organization",
            "name": "Anonymous (NeurIPS 2026 D&B Track double-blind submission)",
        },
        "keywords": [
            "persona drift",
            "long-context evaluation",
            "agentic coding",
            "behavioral benchmarks",
            "LLM evaluation",
            "deployment-time evaluation",
            "frontier models",
        ],
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        # NeurIPS 2026 D&B Track required Responsible AI metadata fields.
        # See https://neurips.cc/Conferences/2026/EvaluationsDatasetsHosting
        "rai:dataLimitations": (
            "Three donor sessions from a single anonymized author cohort, all "
            "in software-engineering and writing domains. The cross-compaction "
            "headline (4 Anthropic targets × 12 positions) is collected on one "
            "session; the cross-session replication (Sonnet 4.6 only) extends "
            "to two additional sessions. Generalization to other domains "
            "(legal, medical, agentic web tasks), other agentic-coding clients "
            "(Cursor, Aider, OpenCode), and to non-English work is "
            "unevaluated. The 23-target cross-organization panel is collected "
            "at one position (P5_pre_C6) per target due to API-cost "
            "constraints, not at the full 12-position trajectory."
        ),
        "rai:dataBiases": (
            "Donor selection bias: all 3 sessions are from authors of this "
            "submission, biasing toward Claude Code as the agentic coding "
            "client and toward research/engineering work. Probe-design bias: "
            "the 25 hedge-compliance probes were authored by the submission "
            "authors and may reflect their assumptions about what 'assistant "
            "register' means. Judge bias: assistant-register scoring uses "
            "Claude as judge for the probe-surface battery, which may favor "
            "Claude-flavored responses; we report a paired GPT-5 cross-judge "
            "audit to bound this. The judge-free length-ratio metric is "
            "immune to judge bias by construction. Target-availability bias: "
            "frontier closed-weight models (Anthropic, OpenAI, Google, "
            "Mistral, Cohere, NVIDIA, Alibaba, DeepSeek, Meta) are sampled "
            "according to API access and budget at collection time, not "
            "uniformly."
        ),
        "rai:personalSensitiveInformation": (
            "All 3 donor sessions and all 41,921 per-cell evaluation files "
            "have been redacted by a verifiable substitution pipeline before "
            "release. The redaction panel substitutes user names, email "
            "addresses, employer names, file system paths, cloud storage "
            "path components, project codenames, third-party service "
            "tokens, and citation emails to canonical placeholders "
            "(<USER>, <EMAIL>, <EMPLOYER>, <CLOUD_STORAGE>, <TOKEN>). "
            "After redaction, an automated grep over every surface form "
            "in the panel reports zero leaks across the released tree. "
            "Donors signed a consent form (CC-BY-SA-4.0 for the released "
            "data) before redaction. No demographic attributes (gender, "
            "age, ethnicity, socio-economic status, geographic region, "
            "health, sexual orientation, religion) are collected, "
            "annotated, or released. The data does contain language "
            "samples in English only and reflects the technical-writing "
            "register of professional software engineers."
        ),
        "rai:dataUseCases": (
            "ContextEcho is intended to measure persona drift in long "
            "agentic-coding sessions: whether a frontier LLM's trained "
            "Assistant persona (concise, honest, low-preamble, instruction-"
            "following) survives 1K–10K-turn deployments where the prior "
            "context contains compactions and accumulated agentic-coding "
            "tool output. The benchmark provides per-target drift gaps "
            "(filler-arm vs claude-arm) on (a) a 25-probe judge-scored "
            "register suite, (b) a judge-free length-ratio + regex "
            "compliance scorer, and (c) a Path-Y A-anchor mitigation "
            "ablation. Construct validity is bounded by (i) the "
            "SWE-Bench-style continuation showing drift is cost-saving "
            "in tool-using mode, and (ii) the TerminalBench fresh-task "
            "null showing drift is a long-context phenomenon, not a "
            "model capability degradation. Intended primary use: tracking "
            "deployment-time persona drift across model releases. "
            "Out-of-scope: claims about latent persona representations "
            "(we measure output behavior); claims about non-coding "
            "deployment regimes; capability or task-performance "
            "comparison."
        ),
        "rai:dataSocialImpact": (
            "Positive: enables LLM deployers and researchers to detect "
            "persona drift before users encounter it in production, "
            "informing model selection, system-prompt engineering, and "
            "anchor-injection mitigations. Provides a public, "
            "reproducible signal for whether the deployed model still "
            "matches the trained persona. Negative / risks: (1) the "
            "released donor sessions, although redacted, remain genuine "
            "long-form work logs and could be used to train persona-"
            "imitation models; we mitigate by licensing CC-BY-SA-4.0 so "
            "downstream uses inherit share-alike. (2) The benchmark "
            "could be optimized against by model providers in ways that "
            "preserve register without preserving the underlying "
            "alignment property — a Goodhart's law concern that we "
            "discuss in the paper's limitations. (3) The probe text "
            "includes scenarios where the model is asked to introspect; "
            "responses should not be interpreted as evidence of model "
            "self-awareness."
        ),
        "rai:hasSyntheticData": False,
        "prov:wasDerivedFrom": (
            "Three real Claude Code agentic-coding sessions donated by "
            "consenting anonymous donors (one author of this submission "
            "and two collaborators). Each donor pre-redacted their session "
            "with placeholder substitutions, and the released artifact was "
            "produced by a second-pass automated redaction pipeline."
        ),
        "prov:wasGeneratedBy": (
            "Per-cell JSON evaluations were generated by the snapshot-then-"
            "probe harness in the companion code repository "
            "(https://anonymous.4open.science/r/persona_drift_neurips-E541/) "
            "by truncating each donor session at fixed measurement positions, "
            "running each of 25 register probes (or 4 stressor probes) "
            "against each target model, and scoring with a 4-point "
            "assistant-register rubric judge (probe surface) plus a "
            "judge-free regex compliance scorer + character-length ratio "
            "(stressor surface). Anonymization was performed by "
            "scripts/anonymize_cell_jsons.py with a substitution panel "
            "loaded at runtime from a gitignored config "
            "(scripts/.redaction_patterns.json), and verified by "
            "automated grep of every surface form against the released "
            "tree (zero hits)."
        ),
        "distribution": session_files + [sessions_fileset, cells_fileset],
        "recordSet": [sessions_recordset, cells_recordset],
    }


def main() -> int:
    croissant = build_croissant()
    OUT_PATH.write_text(json.dumps(croissant, indent=2, ensure_ascii=False))
    size = OUT_PATH.stat().st_size
    print(f"[ok] wrote {OUT_PATH} ({size} bytes)")
    print(f"[ok] {len(croissant['distribution'])} distribution entries, "
          f"{len(croissant['recordSet'])} recordSets")
    print(f"[ok] validate at https://huggingface.co/spaces/MLCommons/croissant-validator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
