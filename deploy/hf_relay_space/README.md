---
title: ContextEcho Donation Relay
emoji: 🧭
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# ContextEcho Donation Relay

Receives already-redacted ContextEcho donation artifacts and forwards them to
private maintainer staging.

Required Space secret:

- `HF_STAGING_TOKEN`

Optional Space variables:

- `CONTEXTECHO_STAGING_REPO=contextecho2026/persona-drift-staging`
- `CONTEXTECHO_RELAY_STATE_DIR=/data/.relay_state`
- `CONTEXTECHO_RELAY_MAX_SESSION_BYTES=209715200`
