"""Deterministic, anonymous donor device identifier.

Donations from the same machine carry the same one-way hash so maintainers
can group a donor's contributions even when the typed name/email/institute
differ between submissions, and so the wizard can show the donor their own
donation history after local receipts or transcripts are lost.

Design constraints:
- DERIVED, never stored: local files can be deleted; the OS machine
  identifier cannot, so the ID is recomputable at any time.
- One-way: only sha256(hardware id + username) ever leaves the machine.
  The raw hardware UUID is never transmitted or written to any artifact.
- Maintainer-visible only: the hash lives in the private staging manifest;
  it is excluded from the public release export, ledger, dataset card,
  and leaderboard.
"""
from __future__ import annotations

import getpass
import hashlib
import re
import subprocess
import sys
from pathlib import Path

_SALT = "contextecho-device-v1"
_UUID_RE = re.compile(r"[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}")


def _macos_hardware_uuid() -> str:
    try:
        out = subprocess.run(
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except Exception:
        return ""
    for line in out.splitlines():
        if "IOPlatformUUID" in line:
            m = _UUID_RE.search(line)
            if m:
                return m.group(0).lower()
    return ""


def _linux_machine_id() -> str:
    for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            value = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value.lower()
    return ""


def _windows_machine_guid() -> str:
    try:
        import winreg  # type: ignore[import-not-found]

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
            value, _ = winreg.QueryValueEx(key, "MachineGuid")
        return str(value).lower()
    except Exception:
        return ""


def hardware_identifier() -> str:
    """Stable per-machine identifier from the OS; empty string if unavailable."""
    if sys.platform == "darwin":
        return _macos_hardware_uuid()
    if sys.platform.startswith("linux"):
        return _linux_machine_id()
    if sys.platform.startswith("win"):
        return _windows_machine_guid()
    return ""


def device_id() -> str:
    """One-way hash identifying this machine+user's donation stream.

    Empty string when no stable hardware identifier is available — callers
    must treat that as "no device linkage", never as a shared bucket.
    """
    hw = hardware_identifier()
    if not hw:
        return ""
    try:
        user = getpass.getuser()
    except Exception:
        user = ""
    digest = hashlib.sha256(f"{_SALT}|{hw}|{user}".encode("utf-8")).hexdigest()
    return digest[:32]


def is_valid_device_id(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{32}", value) is not None


if __name__ == "__main__":
    print(device_id() or "(no stable hardware identifier available)")
