"""Minimal static server for the site/ directory."""
import http.server
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site")
PORT = int(os.environ.get("PORT", 8941))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)


http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
