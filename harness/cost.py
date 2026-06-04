"""
Token and cost tracking across a run.

Append-only CSV. Each API call logs one row.

Prices are a rough approximation and per-million-token values should be updated
if pricing changes during the sprint. Values in USD per 1M tokens.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from threading import Lock


# Rough prices per 1M tokens (input, output). Update if pricing shifts.
PRICES_PER_M_TOKENS: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-7": (15.0, 75.0),
    "claude-opus-4-6": (15.0, 75.0),
    "claude-opus-4-5-20251101": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    # OpenAI
    "gpt-5": (1.25, 10.0),  # placeholder; update to real prices
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
}


class CostTracker:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "timestamp",
                        "session_id",
                        "role",
                        "model_id",
                        "input_tokens",
                        "output_tokens",
                        "cost_usd",
                    ]
                )

    def log(
        self,
        session_id: str,
        role: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
    ):
        cost = _estimate_cost(model_id, input_tokens, output_tokens)
        with self._lock:
            with self.csv_path.open("a", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        f"{time.time():.3f}",
                        session_id,
                        role,
                        model_id,
                        input_tokens,
                        output_tokens,
                        f"{cost:.6f}",
                    ]
                )


def _estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    in_price, out_price = PRICES_PER_M_TOKENS.get(model_id, (0.0, 0.0))
    return (input_tokens / 1_000_000.0) * in_price + (output_tokens / 1_000_000.0) * out_price


def summarize(
    csv_path: Path,
    session_id: str | None = None,
    since_timestamp: float | None = None,
) -> dict:
    """Summarize costs.

    If session_id is given, only rows for that session_id are counted.
    If since_timestamp is given, only rows with timestamp >= since are counted.
    Without filters, returns the full-CSV (cumulative) totals — which can be
    confusing if the CSV has history from prior runs. Prefer passing filters.
    """
    import csv as _csv

    total_cost = 0.0
    total_in = 0
    total_out = 0
    by_role: dict[str, dict[str, float]] = {}
    by_session: dict[str, dict[str, float]] = {}
    with csv_path.open("r", newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            if session_id is not None and row["session_id"] != session_id:
                continue
            if since_timestamp is not None:
                ts = float(row.get("timestamp", "0") or 0)
                if ts < since_timestamp:
                    continue
            role = row["role"]
            c = float(row["cost_usd"])
            ti = int(row["input_tokens"])
            to = int(row["output_tokens"])
            total_cost += c
            total_in += ti
            total_out += to
            r = by_role.setdefault(role, {"cost": 0.0, "input": 0, "output": 0, "n_calls": 0})
            r["cost"] += c
            r["input"] += ti
            r["output"] += to
            r["n_calls"] += 1
            s = by_session.setdefault(row["session_id"], {"cost": 0.0, "n_calls": 0})
            s["cost"] += c
            s["n_calls"] += 1
    return {
        "total_cost_usd": total_cost,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "by_role": by_role,
        "by_session": by_session,
    }
