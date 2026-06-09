# ContextEcho Donor Privacy

ContextEcho studies whether AI assistants drift during long coding sessions. It
does not study donor personality, sentiment, mental health, work performance, or
private beliefs.

The donation wizard supports two privacy tiers:

- `full_redacted` is the default. It keeps the transcript structure and task
  semantics, while removing PII, secrets, paths, URLs, usernames, and donor
  scrub terms. This gives the highest scientific fidelity.
- `user_minimized` is optional. It first performs full redaction, then masks
  donor-authored free text with placeholders and coarse labels. Assistant and
  tool behavior remains available, but analysis detail is lower.

Disallowed uses of donated data include donor profiling, psychological analysis,
sentiment analysis of donors, employment evaluation, and deanonymization.
