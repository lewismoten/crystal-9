#!/usr/bin/env python3
"""Loopback-only static server for the Crystal-9 PNG gallery."""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = ROOT.parent / "browser-runtime"


class GalleryHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def translate_path(self, path: str) -> str:
        """Expose the browser runtime only beneath this loopback gallery origin."""
        request_path = unquote(urlsplit(path).path)
        if request_path == "/runtime" or request_path.startswith("/runtime/"):
            relative = request_path.removeprefix("/runtime").lstrip("/")
            candidate = (RUNTIME_ROOT / relative).resolve()
            try:
                candidate.relative_to(RUNTIME_ROOT.resolve())
            except ValueError:
                return str(RUNTIME_ROOT / "__not_found__")
            return str(candidate)
        return super().translate_path(path)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18846)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), GalleryHandler)
    print(f"Crystal-9 PNG gallery listening on http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
