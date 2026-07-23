#!/usr/bin/env python3
"""PitchProb prediction API.

POST /api/predict  body: {user_id, event_id, hs, as}
  - validates event is pre-kickoff (state from cached feed.json)
  - validates scores are 0-20 integers
  - appends to votes.jsonl
  - 200 {ok:true} or 4xx error

Stateless. Aggregation into crowd.json is handled by pitchprob_feed.py cron.
"""

import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime, timezone

DATA = "/var/www/pitchprob/data"
VOTES = os.path.join(DATA, "votes.jsonl")
FEED = os.path.join(DATA, "feed.json")
UID_RE = re.compile(r"^[a-zA-Z0-9_-]{4,64}$")


def load_event_states():
    try:
        d = json.load(open(FEED))
        return {e["id"]: e["status"]["type"]["state"]
                for e in d.get("scoreboard", {}).get("events", [])}
    except Exception:
        return {}


class API(BaseHTTPRequestHandler):
    def _send(self, status, body):
        b = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/api/health":
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/predict":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 1024:
                return self._send(413, {"error": "too big"})
            body = json.loads(self.rfile.read(n))
        except Exception:
            return self._send(400, {"error": "bad json"})

        uid = str(body.get("user_id", ""))
        eid = str(body.get("event_id", ""))
        if not UID_RE.match(uid):
            return self._send(400, {"error": "invalid user_id"})
        if not eid.isdigit() or len(eid) > 16:
            return self._send(400, {"error": "invalid event_id"})
        try:
            hs = int(body.get("hs")); as_ = int(body.get("as"))
        except Exception:
            return self._send(400, {"error": "scores must be int"})
        if not (0 <= hs <= 20 and 0 <= as_ <= 20):
            return self._send(400, {"error": "scores out of range"})

        states = load_event_states()
        st = states.get(eid)
        if st is None:
            return self._send(404, {"error": "unknown event"})
        if st != "pre":
            return self._send(409, {"error": "match already started"})

        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with open(VOTES, "a") as f:
            f.write(json.dumps({"ts": ts, "uid": uid, "eid": eid,
                                 "hs": hs, "as_": as_}) + "\n")
        return self._send(200, {"ok": True})

    def log_message(self, *_a):
        pass   # silence default request logging


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    ThreadingHTTPServer(("127.0.0.1", 9001), API).serve_forever()
