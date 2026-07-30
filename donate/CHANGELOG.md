# donate/ changelog

## contextecho-donate 0.4 (identity-seeded-redaction) — 2026-07-28

- **Identity-seeded redaction** (`donate/identity_redaction.py`): the wizard
  now uses the consent-form fields it already collects (name, email,
  institute) as scrub input. The engine expands them into the forms donor
  identity actually leaks as — name parts, `First-Last` / `first.last` /
  `firstlast` joins, the email local-part (workspace/handle reuse), institute
  short forms — each mapped to the pipeline's existing salted pseudonyms.
- **Path/slug-aware boundaries:** matches fire inside `/Users/<name>/`,
  `C:\Users\<name>\`, `-Users-<name>-...` slugs, and `<name>-submission`
  labels; a high-entropy guard prevents false positives on short names
  strictly inside base64/hex blobs.
- **Structured-context rules:** git-log `Author:`/`Committer:`/
  `Co-authored-by:` lines and generic `Name <email>` constructs are scrubbed
  whenever the email is the donor's, even if the display name is a nickname
  not in the term set; `user.name=`/`user.email=` git-config lines likewise.
- **JSONL safety:** identity scrubbing parses each line and rewrites string
  values only, so output validity is preserved by construction.
- **Fail-closed verify extension:** `donate/verify.py` gains a blocking
  `donor_identity` category — submission is blocked while any donor identity
  term remains in the redacted artifact (samples are masked in all output).
- **Third-party person review:** the Redact step surfaces a "People detected
  in content" list (PERSON candidates in prose-like fields, e.g. names in
  model-written meeting summaries) with per-item keep/scrub choice,
  defaulting to scrub. Nothing is auto-scrubbed without donor confirmation.
- Tool version bumped: `contextecho-donate 0.3 (multi-agent-discovery)` →
  `contextecho-donate 0.4 (identity-seeded-redaction)`.

## contextecho-donate 0.3 (multi-agent-discovery)

- Prior release (multi-agent session discovery; Presidio + detect-secrets
  pipeline; salted pseudonyms; fail-closed verify).
