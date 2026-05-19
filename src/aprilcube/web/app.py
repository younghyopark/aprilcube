"""Standalone entry point for the browser-only AprilCube designer."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path


def designer_path() -> Path:
    """Return the standalone HTML designer path."""
    return Path(__file__).with_name("static") / "index.html"


def main(argv: list[str] | None = None) -> None:
    """Open or print the static designer; no local server is required."""
    parser = argparse.ArgumentParser(
        description="Open the standalone AprilCube voxel designer.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Print the local HTML path without opening a browser.",
    )
    # Accepted for compatibility with the earlier server-backed prototype.
    parser.add_argument("--host", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--port", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--reload", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    path = designer_path().resolve()
    if not path.exists():
        print(f"Designer file not found: {path}", file=sys.stderr)
        raise SystemExit(1)

    if args.no_open:
        print(path)
        return

    url = path.as_uri()
    if webbrowser.open(url):
        print(f"Opened {url}")
    else:
        print(f"Open this file in your browser: {path}")


if __name__ == "__main__":
    main()
