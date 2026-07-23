#!/usr/bin/env python3
"""PitchProb prediction logger - freezes pre-match predictions for the post-WC study.

Runs from cron every 30 min on the VPS. For every upcoming match within 48h of
kickoff it computes the same prediction the site shows (live Elo + heat/rest/
travel context) and appends a snapshot to predictions.jsonl. Finished matches
are appended once to results.jsonl. Both files are public under /data/.

The last snapshot before kickoff is the "frozen" prediction for the study.
Constants must stay identical to site/index.html.
"""

import json
import math
import os
import urllib.request
from datetime import datetime, timezone

SITE = "/var/www/pitchprob"
DATA = os.path.join(SITE, "data")
STATE_PATH = "/root/pitchprob_logger_state.json"
ESPN = ("https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/"
        "scoreboard?dates=20260611-20260720&limit=300")
KO_START = "2026-06-28"

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
HEAT_T0, HEAT_GOALS, HEAT_GAP = 26.0, 0.006, 0.004
REST_ELO, TRAVEL_ELO = 12.0, 4.0
SNAPSHOT_GAP_H = 6      # min hours between snapshots of the same match
HORIZON_H = 48          # start logging when kickoff is this close


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "pitchprob-logger/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def pmf(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def dc_tau(x, y, lh, la, rho):
    if x == 0 and y == 0: return 1 - lh * la * rho
    if x == 0 and y == 1: return 1 + lh * rho
    if x == 1 and y == 0: return 1 + la * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0


def outcome_probs(lh, la, rho, G=9):
    ph = pd = pa = 0.0
    for x in range(G + 1):
        for y in range(G + 1):
            p = pmf(x, lh) * pmf(y, la) * dc_tau(x, y, lh, la, rho)
            if x > y: ph += p
            elif x == y: pd += p
            else: pa += p
    s = ph + pd + pa
    return ph / s, pd / s, pa / s


def top_scoreline(lh, la, rho, G=6):
    best = None
    for x in range(G + 1):
        for y in range(G + 1):
            p = pmf(x, lh) * pmf(y, la) * dc_tau(x, y, lh, la, rho)
            if best is None or p > best[2]:
                best = (x, y, p)
    return best


def haversine_km(a, b):
    R = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    s = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(s))


def main():
    model = json.load(open(os.path.join(SITE, "wc_model.json")))
    gm = model["goal_map"]
    hosts = set(model["hosts"])
    feed = get_json(ESPN)

    events = []
    for e in feed.get("events", []):
        c = e["competitions"][0]
        home = next(x for x in c["competitors"] if x["homeAway"] == "home")
        away = next(x for x in c["competitors"] if x["homeAway"] == "away")
        if any(k in home["team"]["displayName"] for k in ("Winner", "Place", "Third")):
            continue
        events.append({
            "id": e["id"], "date": e["date"],
            "home": home["team"]["displayName"], "away": away["team"]["displayName"],
            "hs": int(home.get("score") or 0), "as": int(away.get("score") or 0),
            "completed": e["status"]["type"]["completed"],
            "state": e["status"]["type"]["state"],
            "city": (c.get("venue") or {}).get("address", {}).get("city"),
        })
    events.sort(key=lambda x: x["date"])

    # live Elo replay (same rule as the site)
    elo = dict(model["elo"])
    for ev in events:
        if not ev["completed"] or ev["home"] not in elo or ev["away"] not in elo:
            continue
        adv = (model["home_adv_elo"] if ev["home"] in hosts else 0) \
            - (model["home_adv_elo"] if ev["away"] in hosts else 0)
        exp = 1 / (1 + 10 ** (-((elo[ev["home"]] + adv - elo[ev["away"]]) / 400)))
        s = 1.0 if ev["hs"] > ev["as"] else 0.5 if ev["hs"] == ev["as"] else 0.0
        d = abs(ev["hs"] - ev["as"])
        mult = 1.0 if d <= 1 else 1.5 if d == 2 else (11 + d) / 8
        delta = model["elo_k_wc"] * mult * (s - exp)
        elo[ev["home"]] += delta
        elo[ev["away"]] -= delta

    os.makedirs(DATA, exist_ok=True)
    state = {}
    if os.path.exists(STATE_PATH):
        state = json.load(open(STATE_PATH))
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat(timespec="seconds")

    # ---- results: append once per completed match ----
    logged_results = set(state.get("results", []))
    with open(os.path.join(DATA, "results.jsonl"), "a") as f:
        for ev in events:
            if ev["completed"] and ev["id"] not in logged_results:
                f.write(json.dumps({"logged_at": now_iso, "event_id": ev["id"],
                                    "kickoff": ev["date"], "home": ev["home"], "away": ev["away"],
                                    "score": [ev["hs"], ev["as"]],
                                    "outcome": "H" if ev["hs"] > ev["as"] else "D" if ev["hs"] == ev["as"] else "A"}) + "\n")
                logged_results.add(ev["id"])
    state["results"] = sorted(logged_results)

    # ---- weather, fetched once per relevant city ----
    upcoming = [ev for ev in events if ev["state"] == "pre" and ev["home"] in elo
                and 0 <= (datetime.fromisoformat(ev["date"].replace("Z", "+00:00")) - now).total_seconds() <= HORIZON_H * 3600]
    wx = {}
    for city in {ev["city"] for ev in upcoming if ev["city"] in VENUES}:
        try:
            lat, lon = VENUES[city]
            wx[city] = get_json(
                f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                f"&hourly=apparent_temperature,precipitation&timezone=UTC&forecast_days=16")["hourly"]
        except Exception:
            pass

    def last_appearance(team, before):
        last = None
        for ev in events:
            if ev["completed"] and ev["date"] < before and team in (ev["home"], ev["away"]):
                last = ev
        return last

    # ---- prediction snapshots ----
    snaps = state.get("snapshots", {})
    n_new = 0
    with open(os.path.join(DATA, "predictions.jsonl"), "a") as f:
        for ev in upcoming:
            last_snap = snaps.get(ev["id"])
            if last_snap and (now - datetime.fromisoformat(last_snap)).total_seconds() < SNAPSHOT_GAP_H * 3600:
                continue
            adv = (model["home_adv_elo"] if ev["home"] in hosts else 0) \
                - (model["home_adv_elo"] if ev["away"] in hosts else 0)
            elo_adj, gap_mult, goals_mult, notes = 0.0, 1.0, 1.0, []
            temp_c = precip = None
            h = wx.get(ev["city"])
            if h:
                ko = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
                key = ko.strftime("%Y-%m-%dT%H:00")
                if key in h["time"]:
                    i = h["time"].index(key)
                    temp_c, precip = h["apparent_temperature"][i], h["precipitation"][i]
                    if temp_c is not None and temp_c > HEAT_T0:
                        goals_mult *= clamp(1 - HEAT_GOALS * (temp_c - HEAT_T0), 0.88, 1)
                        gap_mult *= clamp(1 - HEAT_GAP * (temp_c - HEAT_T0), 0.92, 1)
                        notes.append(f"heat:{temp_c}")
                    if precip is not None and precip >= 1:
                        goals_mult *= 0.98
                        notes.append("rain")
            lh_m, la_m = last_appearance(ev["home"], ev["date"]), last_appearance(ev["away"], ev["date"])
            if lh_m and la_m:
                days = lambda m: (datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
                                  - datetime.fromisoformat(m["date"].replace("Z", "+00:00"))).total_seconds() / 86400
                rest_diff = days(lh_m) - days(la_m)
                if abs(rest_diff) >= 0.75:
                    elo_adj += clamp(REST_ELO * rest_diff, -25, 25)
                    notes.append(f"rest:{rest_diff:+.1f}")
                if ev["city"] in VENUES and lh_m["city"] in VENUES and la_m["city"] in VENUES:
                    tdiff = (haversine_km(VENUES[lh_m["city"]], VENUES[ev["city"]])
                             - haversine_km(VENUES[la_m["city"]], VENUES[ev["city"]]))
                    if abs(tdiff) > 800:
                        elo_adj += clamp(-TRAVEL_ELO * tdiff / 1000, -12, 12)
                        notes.append(f"travel:{tdiff:+.0f}km")
            x = (elo[ev["home"]] + adv + elo_adj - elo[ev["away"]]) / 400 * gap_mult
            lam_h = math.exp(gm["a"] + gm["b"] * x) * goals_mult
            lam_a = math.exp(gm["a"] - gm["b"] * x) * goals_mult
            ph, pd, pa = outcome_probs(lam_h, lam_a, gm["rho"])
            ts = top_scoreline(lam_h, lam_a, gm["rho"])
            pick = "H" if ph >= pd and ph >= pa else "A" if pa >= ph and pa >= pd else "D"
            f.write(json.dumps({
                "logged_at": now_iso, "event_id": ev["id"], "kickoff": ev["date"],
                "home": ev["home"], "away": ev["away"], "city": ev["city"],
                "elo": [round(elo[ev["home"]], 1), round(elo[ev["away"]], 1)],
                "context": {"temp_c": temp_c, "precip_mm": precip, "elo_adj": round(elo_adj, 1),
                            "gap_mult": round(gap_mult, 4), "goals_mult": round(goals_mult, 4),
                            "notes": notes},
                "lambda": [round(lam_h, 3), round(lam_a, 3)],
                "probs": {"H": round(ph, 4), "D": round(pd, 4), "A": round(pa, 4)},
                "pick": pick, "top_score": [ts[0], ts[1]],
            }) + "\n")
            snaps[ev["id"]] = now_iso
            n_new += 1
    state["snapshots"] = snaps
    json.dump(state, open(STATE_PATH, "w"))
    print(f"{now_iso} snapshots+{n_new} results={len(logged_results)} upcoming={len(upcoming)}")


if __name__ == "__main__":
    main()
