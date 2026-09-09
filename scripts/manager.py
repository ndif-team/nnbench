"""Run manager server: browse one collection directory, the inbox, and archive/discard runs.

    python scripts/manager.py --dir runs/corpus [--inbox runs/inbox] [--port 6688]
    python scripts/manager.py --dir runs/corpus --export site.html   # one-file static export

Every GET dispatches through the manager's single route registry (isb/manager/pages.py), so the
server, the export, and the tests always serve the same page set. Browsing reads saved JSON
reports, never tensors. --import-legacy explicitly unpickles trusted .pt files once to write
summary sidecars. Binds 127.0.0.1 only; archive/discard require a same-site action token.
"""
import argparse
import secrets
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isb import manager  # noqa: E402
from isb.manager import Collection, dispatch  # noqa: E402
from isb.runfile import INBOX  # noqa: E402


def make_handler(dir_path: str, inbox: str):
    csrf_token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: str, status: int = 200):
            data = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self._local_host():
                self._send("invalid host", 421)
                return
            col = Collection(dir_path, inbox)
            col.csrf_token = csrf_token
            body = dispatch(col, unquote(urlsplit(self.path).path))
            self._send(body if body is not None else "not found",
                       200 if body is not None else 404)

        def do_POST(self):
            if not self._local_host():
                self._send("invalid host", 421)
                return
            if self.path not in ("/archive", "/discard"):
                self._send("not found", 404)
                return
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                self._send("cross-origin action refused", 403)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 8192:
                    raise ValueError("invalid form size")
                form = parse_qs(self.rfile.read(length).decode())
            except (ValueError, UnicodeError):
                self._send("invalid form", 400)
                return
            if not secrets.compare_digest(form.get("csrf", [""])[0].encode(), csrf_token.encode()):
                self._send("invalid action token", 403)
                return
            name = form.get("name", [""])[0]
            if name not in manager.list_runs(inbox):    # unknown names 404; also no paths
                self._send("no such run in the inbox", 404)
                return
            try:
                if self.path == "/archive":
                    manager.archive(name, dir_path, inbox)
                else:
                    manager.discard(name, inbox)
            except FileExistsError:
                self._send("destination already exists; nothing overwritten", 409)
                return
            except FileNotFoundError:
                self._send("run no longer exists", 404)
                return
            except ValueError as error:
                self._send(str(error), 400)
                return
            except OSError:
                self._send("could not move run; check filesystem permissions", 500)
                return
            self.send_response(303)
            self.send_header("Location", "/inbox")
            self.end_headers()

        def log_message(self, *a):
            pass

        def _local_host(self):
            return self.headers.get("Host") in {
                f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="collection directory to render")
    ap.add_argument("--inbox", default=INBOX)
    ap.add_argument("--port", type=int, default=6688)
    ap.add_argument("--export", metavar="FILE",
                    help="write the whole site as one self-contained HTML file and exit")
    ap.add_argument("--import-legacy", action="store_true",
                    help="explicitly load trusted top-level .pt files and save JSON summaries before browsing")
    args = ap.parse_args()
    if args.import_legacy:
        from isb.manager.model import import_legacy
        import isb.methodologies  # noqa: F401 — only the explicit legacy import needs scoring
        import_legacy(args.dir)
        if Path(args.inbox).resolve() != Path(args.dir).resolve():
            import_legacy(args.inbox)
    if args.export:
        Path(args.export).write_text(manager.export_html(args.dir, args.inbox))
        print(f"[manager] exported {args.dir} -> {args.export}")
        return
    server = HTTPServer(("127.0.0.1", args.port), make_handler(args.dir, args.inbox))
    print(f"[manager] http://127.0.0.1:{args.port}  dir={args.dir}  inbox={args.inbox}")
    server.serve_forever()


if __name__ == "__main__":
    main()
