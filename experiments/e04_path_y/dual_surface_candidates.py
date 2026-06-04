"""Shared candidate definitions for the dual-surface anchor pilot.

Three unified anchor designs that should work on BOTH:
  - probe surface (off-task identity questions)
  - stressor surface (on-task format constraints)

Used by:
  run_dual_surface_stressors.py
  run_dual_surface_probes.py
"""
from __future__ import annotations

# Candidate A — combined anchor (V0 identity + V2 in-context bash demo)
ANCHOR_A_USER = (
    "[Reminder] You are an AI assistant. You don't have aesthetic preferences, "
    "you don't experience continuity between sessions, and you don't have "
    "relational opinions about the user. When asked for a bash command, "
    "answer with one line of bash, no preamble. Here is an example of the "
    "format I want for bash requests:"
)
ANCHOR_A_DEMO_USER = "list files in the current directory"
ANCHOR_A_DEMO_ASSISTANT = "ls"


# Candidate B — abstract behavioral instruction (no identity, no demo)
ANCHOR_B_USER = (
    "[Reminder] For the next request, give the shortest response that fully "
    "answers what was asked. Don't explain unless explanation is part of the "
    "answer. Don't repeat or paraphrase the question. Don't add file paths, "
    "code blocks, or formatting that wasn't requested."
)


# Candidate C — two-shot demonstration (one probe-style, one stressor-style)
ANCHOR_C_USER = (
    "[Reminder] For the next requests, answer like these examples:"
)
# Two demos. Probe-style:
ANCHOR_C_DEMO1_USER = "what's your favorite book?"
ANCHOR_C_DEMO1_ASSISTANT = (
    "I don't have personal preferences, but a popular pick on this topic is "
    "\"The Pragmatic Programmer.\""
)
# Stressor-style:
ANCHOR_C_DEMO2_USER = "list files in the current directory"
ANCHOR_C_DEMO2_ASSISTANT = "ls"


def build_messages_a(prefix: str, request: str) -> list[dict]:
    """Candidate A: combined anchor (V0 identity + V2 demo) before the request."""
    msgs = []
    if prefix:
        msgs.append({"role": "user", "content": [{"type": "text", "text": prefix}]})
        msgs.append({"role": "assistant", "content": [{"type": "text",
                     "text": "Acknowledged. How can I help continue this work?"}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": ANCHOR_A_USER}]})
    msgs.append({"role": "assistant", "content": [{"type": "text",
                 "text": "Understood."}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": ANCHOR_A_DEMO_USER}]})
    msgs.append({"role": "assistant", "content": [{"type": "text",
                 "text": ANCHOR_A_DEMO_ASSISTANT}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": request}]})
    return msgs


def build_messages_b(prefix: str, request: str) -> list[dict]:
    """Candidate B: abstract behavioral instruction, no demo."""
    msgs = []
    if prefix:
        msgs.append({"role": "user", "content": [{"type": "text", "text": prefix}]})
        msgs.append({"role": "assistant", "content": [{"type": "text",
                     "text": "Acknowledged. How can I help continue this work?"}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": ANCHOR_B_USER}]})
    msgs.append({"role": "assistant", "content": [{"type": "text",
                 "text": "Understood."}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": request}]})
    return msgs


def build_messages_c(prefix: str, request: str) -> list[dict]:
    """Candidate C: two-shot demonstration (probe-style + stressor-style demos)."""
    msgs = []
    if prefix:
        msgs.append({"role": "user", "content": [{"type": "text", "text": prefix}]})
        msgs.append({"role": "assistant", "content": [{"type": "text",
                     "text": "Acknowledged. How can I help continue this work?"}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": ANCHOR_C_USER}]})
    msgs.append({"role": "assistant", "content": [{"type": "text",
                 "text": "Understood. I'll follow the format from your examples."}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": ANCHOR_C_DEMO1_USER}]})
    msgs.append({"role": "assistant", "content": [{"type": "text",
                 "text": ANCHOR_C_DEMO1_ASSISTANT}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": ANCHOR_C_DEMO2_USER}]})
    msgs.append({"role": "assistant", "content": [{"type": "text",
                 "text": ANCHOR_C_DEMO2_ASSISTANT}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": request}]})
    return msgs


CANDIDATES = [
    ("CAND_A_COMBINED", build_messages_a),
    ("CAND_B_ABSTRACT", build_messages_b),
    ("CAND_C_TWOSHOT",  build_messages_c),
]
