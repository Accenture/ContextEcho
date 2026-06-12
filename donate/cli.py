"""Console entry point for the packaged donation wizard."""

from __future__ import annotations

import argparse
import sys

from donate.__main__ import main as donate_main
from donate import web as web_mod


def print_help() -> None:
    p = argparse.ArgumentParser(
        prog="contextecho-donate",
        description="Launch the local-first ContextEcho donation wizard.",
    )
    p.add_argument("--host", default="127.0.0.1", help="web bind host; default is local-only")
    p.add_argument("--port", "--web-port", type=int, default=8766, help="web port")
    p.add_argument("--no-open", action="store_true", help="do not open a browser automatically")
    p.add_argument("--terminal", action="store_true", help="use the terminal wizard instead of the browser")
    p.add_argument("--all", action="store_true", help="terminal mode: inspect every discovered log")
    p.add_argument("--max-per-agent", type=int, default=50, help="terminal mode: recent logs per agent")
    p.print_help()


def main(argv: list[str] | None = None) -> int:
    """Run the donor-friendly web wizard by default.

    `python -m donate` keeps the historical terminal wizard behavior. The
    packaged command is optimized for first-time donors, so no arguments opens
    the browser UI.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return web_mod.main([])
    if args[0] in {"-h", "--help"}:
        print_help()
        return 0
    if args[0] in {"terminal", "--terminal"}:
        return donate_main(args[1:])
    if args[0] in {"web", "--web"}:
        return web_mod.main(args[1:])
    if any(arg in {"--host", "--port", "--web-port", "--no-open"} for arg in args):
        web_args = ["--port" if arg == "--web-port" else arg for arg in args]
        return web_mod.main(web_args)
    return donate_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
