"""Run manager server: browse one collection directory, the inbox, and archive/discard runs.

    python scripts/manager.py --dir runs/corpus [--inbox runs/inbox] [--port 6688]
    python scripts/manager.py --dir runs/corpus --export site.html   # one-file static export

Every GET dispatches through the manager's single route registry (isb/manager/pages.py), so the
server, the export, and the tests always serve the same page set. Binds 127.0.0.1 only: run
files are unpickled on load, so the manager serves only files you produced locally, to you.
"""
import argparse
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import isb.methodologies  # noqa: F401,E402  (registers the cells for scoring)
from isb import manager  # noqa: E402
from isb.manager import Collection, dispatch  # noqa: E402
from isb.runfile import INBOX  # noqa: E402


def make_handler(dir_path: str, inbox: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: str, status: int = 200):
            data = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            body = dispatch(Collection(dir_path, inbox), unquote(self.path))
            self._send(body if body is not None else "not found",
                       200 if body is not None else 404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(length).decode())
            name = form.get("name", [""])[0]
            if name not in manager.list_runs(inbox):    # unknown names 404; also no paths
                self._send("no such run in the inbox", 404)
                return
            if self.path == "/archive":
                manager.archive(name, dir_path, inbox)
            elif self.path == "/discard":
                manager.discard(name, inbox)
            self.send_response(303)
            self.send_header("Location", "/inbox")
            self.end_headers()

        def log_message(self, *a):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="collection directory to render")
    ap.add_argument("--inbox", default=INBOX)
    ap.add_argument("--port", type=int, default=6688)
    ap.add_argument("--export", metavar="FILE",
                    help="write the whole site as one self-contained HTML file and exit")
    args = ap.parse_args()
    if args.export:
        Path(args.export).write_text(manager.export_html(args.dir, args.inbox))
        print(f"[manager] exported {args.dir} -> {args.export}")
        return
    server = HTTPServer(("127.0.0.1", args.port), make_handler(args.dir, args.inbox))
    print(f"[manager] http://127.0.0.1:{args.port}  dir={args.dir}  inbox={args.inbox}")
    server.serve_forever()


if __name__ == "__main__":
    main()
