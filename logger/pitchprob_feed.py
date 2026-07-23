#!/usr/bin/env python3
"""PitchProb feed cache - runs every minute from cron.

Fetches the ESPN scoreboard once and writes it to the public data/ dir so
every site visitor reads our cached copy instead of hitting ESPN themselves
(N visitors -> 1 upstream request/min). Weather forecasts for the 16 host
cities are refreshed every ~25 minutes into weather.json the same way.
Writes are atomic (tmp + rename) so visitors never see a torn file.
"""

import json
import os
import time
import urllib.request
from datetime import datetime, timezone

DATA = "/var/www/pitchprob/data"
ESPN = ("https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/"
        "scoreboard?dates=20260611-20260720&limit=300")
WEATHER_MAX_AGE_S = 25 * 60

VENUES = {
    "Mexico City": (19.30, -99.15), "Guadalajara": (20.68, -103.46), "Guadalupe": (25.67, -100.24),
    "Toronto": (43.63, -79.42), "Vancouver": (49.28, -123.11),
    "Seattle, Washington": (47.60, -122.33), "Santa Clara, California": (37.40, -121.97),
    "Inglewood, California": (33.95, -118.34), "Kansas City, Missouri": (39.05, -94.48),
    "Arlington, Texas": (32.75, -97.09), "Houston, Texas": (29.68, -95.41),
    "Atlanta, Georgia": (33.75, -84.40), "Miami Gardens, Florida": (25.96, -80.24),
    "Philadelphia, Pennsylvania": (39.90, -75.17), "East Rutherford, New Jersey": (40.81, -74.07),
    "Foxborough, Massachusetts": (42.09, -71.26),
}


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "pitchprob-feed/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def write_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.rename(tmp, path)


def main():
    os.makedirs(DATA, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # scoreboard: every run
    sb = get_json(ESPN)
    if sb.get("events"):  # never overwrite a good file with an empty response
        write_atomic(os.path.join(DATA, "feed.json"),
                     {"fetched_at": now_iso, "scoreboard": sb})

    # rich boxscore (passes, saves, tackles...) for matches currently in play
    by_event = {}
    for e in sb.get("events", []):
        if e["status"]["type"]["state"] != "in":
            continue
        try:
            summ = get_json("https://site.api.espn.com/apis/site/v2/sports/soccer/"
                            f"fifa.world/summary?event={e['id']}")
            comp = e["competitions"][0]
            home_id = next(c["team"]["id"] for c in comp["competitors"] if c["homeAway"] == "home")
            sides = {}
            for t in summ.get("boxscore", {}).get("teams", []):
                side = "home" if t["team"]["id"] == home_id else "away"
                sides[side] = {s["name"]: s.get("displayValue") for s in t.get("statistics", [])}
            if sides:
                by_event[e["id"]] = sides
        except Exception:
            pass
    write_atomic(os.path.join(DATA, "live_stats.json"),
                 {"fetched_at": now_iso, "byEvent": by_event})

    # ---- aggregate user predictions into crowd.json ----
    votes_path = os.path.join(DATA, "votes.jsonl")
    if os.path.exists(votes_path):
        from collections import Counter
        latest = {}  # (uid, eid) -> latest vote
        with open(votes_path) as f:
            for line in f:
                try:
                    v = json.loads(line)
                    latest[(v["uid"], v["eid"])] = v
                except Exception:
                    pass
        by_event_picks = {}
        for v in latest.values():
            by_event_picks.setdefault(v["eid"], []).append((int(v["hs"]), int(v["as_"])))
        crowd = {}
        for eid, picks in by_event_picks.items():
            n = len(picks)
            H = sum(1 for h, a in picks if h > a)
            D = sum(1 for h, a in picks if h == a)
            A = n - H - D
            top_score, top_n = Counter(picks).most_common(1)[0]
            crowd[eid] = {
                "n": n,
                "avg": [round(sum(h for h, _ in picks) / n, 2),
                         round(sum(a for _, a in picks) / n, 2)],
                "picks": {"H": H, "D": D, "A": A},
                "modal_score": list(top_score),
                "modal_n": top_n,
            }
        write_atomic(os.path.join(DATA, "crowd.json"),
                     {"fetched_at": now_iso, "byEvent": crowd})

    # weather: only when stale
    wpath = os.path.join(DATA, "weather.json")
    if not os.path.exists(wpath) or time.time() - os.path.getmtime(wpath) > WEATHER_MAX_AGE_S:
        by_city = {}
        for city, (lat, lon) in VENUES.items():
            try:
                by_city[city] = get_json(
                    f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                    f"&hourly=apparent_temperature,precipitation&timezone=UTC"
                    f"&forecast_days=16&past_days=1")["hourly"]
            except Exception:
                pass
        if by_city:
            write_atomic(wpath, {"fetched_at": now_iso, "byCity": by_city})


if __name__ == "__main__":
    main()
