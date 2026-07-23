"""Build World Cup 2026 prediction model.

Sources:
  data/intl_results.csv      - all international matches 1872-present (martj42/international_results)
  data/espn_standings.json   - WC2026 groups A-L (ESPN API)
  data/espn_knockout.json    - WC2026 knockout placeholders -> bracket structure (ESPN API)

Model:
  - Importance-weighted Elo over full history (margin-of-victory multiplier,
    home advantage, no decay needed since Elo is sequential).
  - Poisson goal mapping: lambda = exp(a + b * elo_diff/400), fit on recent matches.
  - Walk-forward backtest on the last ~2.5 years of internationals.

Output: site/wc_model.json
"""

import csv
import json
import math
from collections import defaultdict

ELO_START = 1500.0
HOME_ADV = 80.0  # Elo points when not on neutral ground


def k_factor(tournament):
    t = tournament.lower()
    if t == "fifa world cup":
        return 60.0
    if "world cup qualification" in t:
        return 50.0
    if t in ("uefa euro", "copa américa", "copa america", "african cup of nations",
             "afc asian cup", "gold cup", "concacaf championship"):
        return 50.0
    if "qualification" in t or "nations league" in t:
        return 40.0
    if t == "friendly":
        return 20.0
    return 30.0


def margin_mult(diff):
    if diff <= 1:
        return 1.0
    if diff == 2:
        return 1.5
    return (11.0 + diff) / 8.0


def load():
    rows = []
    with open("data/intl_results.csv") as f:
        for r in csv.DictReader(f):
            if r["home_score"] == "NA" or not r["home_score"]:
                continue
            rows.append({
                "date": r["date"],
                "home": r["home_team"], "away": r["away_team"],
                "hg": int(float(r["home_score"])), "ag": int(float(r["away_score"])),
                "tournament": r["tournament"],
                "neutral": r["neutral"] == "TRUE",
            })
    rows.sort(key=lambda r: r["date"])
    return rows


def expected(r_home, r_away, neutral):
    adv = 0.0 if neutral else HOME_ADV
    return 1.0 / (1.0 + 10 ** (-(r_home + adv - r_away) / 400.0))


def run_elo(matches, ratings=None):
    ratings = ratings if ratings is not None else {}
    for m in matches:
        rh = ratings.setdefault(m["home"], ELO_START)
        ra = ratings.setdefault(m["away"], ELO_START)
        e = expected(rh, ra, m["neutral"])
        s = 1.0 if m["hg"] > m["ag"] else (0.5 if m["hg"] == m["ag"] else 0.0)
        delta = k_factor(m["tournament"]) * margin_mult(abs(m["hg"] - m["ag"])) * (s - e)
        ratings[m["home"]] = rh + delta
        ratings[m["away"]] = ra - delta
    return ratings


def fit_goal_map(matches, elo_snapshots):
    """Fit lambda = exp(a + b*x), x = (own_elo + adv - opp_elo)/400, by Poisson MLE.

    elo_snapshots[i] = (elo_home, elo_away) at the time of match i.
    Two observations per match (home goals, away goals).
    """
    obs = []
    for m, (rh, ra) in zip(matches, elo_snapshots):
        adv = 0.0 if m["neutral"] else HOME_ADV
        obs.append(((rh + adv - ra) / 400.0, m["hg"]))
        obs.append(((ra - rh - adv) / 400.0, m["ag"]))
    a, b = 0.25, 0.4
    lr = 0.05
    n = len(obs)
    for _ in range(400):
        ga = gb = 0.0
        for x, y in obs:
            lam = math.exp(a + b * x)
            ga += (y - lam)
            gb += (y - lam) * x
        a += lr * ga / n
        b += lr * gb / n
    return a, b


def dc_tau(x, y, lh, la, rho):
    """Dixon-Coles low-score correlation adjustment."""
    if x == 0 and y == 0:
        return 1 - lh * la * rho
    if x == 0 and y == 1:
        return 1 + lh * rho
    if x == 1 and y == 0:
        return 1 + la * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def match_probs(lh, la, rho=0.0, max_goals=10):
    ph = pd = pa = 0.0
    pmf = lambda k, lam: math.exp(-lam) * lam ** k / math.factorial(k)
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            p = pmf(x, lh) * pmf(y, la) * dc_tau(x, y, lh, la, rho)
            if x > y:
                ph += p
            elif x == y:
                pd += p
            else:
                pa += p
    s = ph + pd + pa
    return ph / s, pd / s, pa / s


def backtest(matches, a, b, rho=0.0, home_adv=None, start_date="2024-01-01"):
    home_adv = HOME_ADV if home_adv is None else home_adv
    ratings = {}
    n = correct = 0
    brier = 0.0
    for m in matches:
        if m["date"] >= start_date and m["home"] in ratings and m["away"] in ratings:
            rh, ra = ratings[m["home"]], ratings[m["away"]]
            adv = 0.0 if m["neutral"] else home_adv
            lh = math.exp(a + b * (rh + adv - ra) / 400.0)
            la = math.exp(a + b * (ra - rh - adv) / 400.0)
            ph, pd, pa = match_probs(lh, la, rho)
            pick = max(((ph, "H"), (pd, "D"), (pa, "A")))[1]
            actual = "H" if m["hg"] > m["ag"] else ("D" if m["hg"] == m["ag"] else "A")
            n += 1
            correct += pick == actual
            brier += ((ph - (actual == "H")) ** 2 + (pd - (actual == "D")) ** 2
                      + (pa - (actual == "A")) ** 2)
        run_elo([m], ratings)
    return n, correct / n, brier / n


def tune(matches):
    """Grid-search HOME_ADV, goal-map fit window, and Dixon-Coles rho.

    Selected by Brier score on the 2024+ walk-forward backtest.
    Mutates the global HOME_ADV; returns (a, b, rho, results_table).
    """
    global HOME_ADV
    table = []
    best = None
    for home_adv in (60.0, 80.0, 100.0):
        HOME_ADV = home_adv
        for fit_start in ("2015-01-01", "2018-01-01"):
            ratings = {}
            snapshots, fit_matches = [], []
            for m in matches:
                if m["date"] >= fit_start and m["home"] in ratings and m["away"] in ratings:
                    fit_matches.append(m)
                    snapshots.append((ratings[m["home"]], ratings[m["away"]]))
                run_elo([m], ratings)
            a, b = fit_goal_map(fit_matches, snapshots)
            for rho in (0.0, -0.04, -0.08, -0.12):
                n, acc, br = backtest(matches, a, b, rho, home_adv)
                table.append((home_adv, fit_start[:4], rho, acc, br))
                if best is None or br < best[0]:
                    best = (br, acc, home_adv, fit_start, rho, a, b)
    br, acc, home_adv, fit_start, rho, a, b = best
    HOME_ADV = home_adv
    return a, b, rho, best, table


# ---- name alignment: dataset names -> ESPN displayNames ----
DATASET_TO_ESPN = {
    "Czech Republic": "Czechia",
    "Turkey": "Türkiye",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "DR Congo": "Congo DR",
}


def main():
    matches = load()
    print(f"Loaded {len(matches)} matches, last: {matches[-1]['date']}")

    print("Tuning (home_adv x fit-window x rho), selecting by walk-forward Brier...")
    a, b, rho, best, table = tune(matches)
    for ha, win, r, acc_, br_ in table:
        mark = " <-- best" if (ha, f"{win}-01-01", r) == (best[2], best[3], best[4]) else ""
        print(f"  adv={ha:5.0f} fit>={win} rho={r:+.2f}: acc {acc_:.1%}  brier {br_:.4f}{mark}")
    print(f"Chosen: home_adv={HOME_ADV}, rho={rho}, lambda = exp({a:.4f} + {b:.4f}*diff/400)")

    # rebuild final ratings with chosen home_adv
    ratings = run_elo(matches)
    n, acc, br = backtest(matches, a, b, rho)
    print(f"Final walk-forward: {n} matches, accuracy {acc:.1%}, brier {br:.3f}")

    # groups from ESPN standings
    stand = json.load(open("data/espn_standings.json"))
    groups = {}
    espn_teams = set()
    for ch in stand["children"]:
        letter = ch["name"].replace("Group ", "")
        names = [e["team"]["displayName"] for e in ch["standings"]["entries"]]
        groups[letter] = names
        espn_teams.update(names)

    # final pre-tournament Elo for the 48 teams, keyed by ESPN name
    elo_out = {}
    for ds_name, r in ratings.items():
        espn = DATASET_TO_ESPN.get(ds_name, ds_name)
        if espn in espn_teams:
            elo_out[espn] = round(r, 1)
    missing = espn_teams - set(elo_out)
    if missing:
        raise SystemExit(f"Missing Elo for: {missing}")

    # recent form (last 5) per team, keyed by ESPN name
    form = {}
    for espn in espn_teams:
        ds = next((k for k, v in DATASET_TO_ESPN.items() if v == espn), espn)
        res = []
        for m in reversed(matches):
            if m["home"] == ds:
                res.append("W" if m["hg"] > m["ag"] else ("D" if m["hg"] == m["ag"] else "L"))
            elif m["away"] == ds:
                res.append("W" if m["ag"] > m["hg"] else ("D" if m["hg"] == m["ag"] else "L"))
            if len(res) == 5:
                break
        form[espn] = res

    # all-time head-to-head among the 48 teams (last 6 meetings per pair)
    espn_name = lambda ds: DATASET_TO_ESPN.get(ds, ds)
    h2h = defaultdict(list)
    for m in matches:
        h, aw = espn_name(m["home"]), espn_name(m["away"])
        if h in espn_teams and aw in espn_teams:
            key = "|".join(sorted([h, aw]))
            h2h[key].append([m["date"], h, m["hg"], m["ag"], aw])
    h2h = {k: v[-6:] for k, v in h2h.items()}

    # recent scoring profile: avg goals for / against over each team's last 12 games
    goals = {}
    for espn in espn_teams:
        ds = next((k for k, v in DATASET_TO_ESPN.items() if v == espn), espn)
        gf = ga = nseen = 0
        for m in reversed(matches):
            if m["home"] == ds:
                gf += m["hg"]; ga += m["ag"]; nseen += 1
            elif m["away"] == ds:
                gf += m["ag"]; ga += m["hg"]; nseen += 1
            if nseen == 12:
                break
        goals[espn] = [round(gf / nseen, 2), round(ga / nseen, 2)] if nseen else [0, 0]

    # group-stage fixtures from dataset (future WC matches), mapped to ESPN names
    fixtures = []
    with open("data/intl_results.csv") as f:
        for r in csv.DictReader(f):
            if r["tournament"] == "FIFA World Cup" and r["date"] >= "2026-06-01":
                fixtures.append({
                    "date": r["date"],
                    "home": DATASET_TO_ESPN.get(r["home_team"], r["home_team"]),
                    "away": DATASET_TO_ESPN.get(r["away_team"], r["away_team"]),
                    "city": r["city"],
                })
    print(f"Group-stage fixtures: {len(fixtures)}")

    # knockout bracket from ESPN placeholders, in chronological order
    ko = json.load(open("data/espn_knockout.json"))
    evs = sorted(ko["events"], key=lambda e: e["date"])
    r32, r16, qf, sf, final, third = [], [], [], [], None, None
    for e in evs:
        comp = e["competitions"][0]
        slots = []
        for c in sorted(comp["competitors"], key=lambda c: c["homeAway"] != "home"):
            short = c["team"]["shortDisplayName"]  # e.g. "1C", "2A", "3RD A/B/C/D/F", "RD32 W3"
            name = c["team"]["displayName"]
            if short.startswith("3RD"):
                pools = name.split("Group ")[-1].split("/")
                slots.append({"type": "third", "pool": pools})
            elif short.startswith("RD32"):
                slots.append({"type": "w32", "idx": int(short.split("W")[1]) - 1})
            elif short.startswith("RD16"):
                slots.append({"type": "w16", "idx": int(short.split("W")[1]) - 1})
            elif short.startswith("QF"):
                slots.append({"type": "wqf", "idx": int(short.split("W")[1]) - 1})
            elif short.startswith("SF"):
                kind = "wsf" if "W" in short else "lsf"
                slots.append({"type": kind, "idx": int(short[4:]) - 1})
            else:  # "1C" / "2A"
                slots.append({"type": "group", "rank": int(short[0]), "group": short[1]})
        pair = {"date": e["date"], "slots": slots}
        nm = e["name"]
        if "Round of 32" in nm:
            r16.append(pair)
        elif "Round of 16" in nm:
            qf.append(pair)
        elif "Quarterfinal" in nm:
            sf.append(pair)
        elif "Loser" in nm:
            third = pair
        elif "Semifinal" in nm:
            final = pair
        else:
            r32.append(pair)
    print(f"Bracket: R32={len(r32)} R16={len(r16)} QF={len(qf)} SF={len(sf)}")
    assert len(r32) == 16 and len(r16) == 8 and len(qf) == 4 and len(sf) == 2

    model = {
        "generated": matches[-1]["date"],
        "sources": ["github.com/martj42/international_results (49k matches, 1872-2026)",
                     "ESPN scoreboard API (live)", "ESPN standings API (groups)"],
        "goal_map": {"a": round(a, 4), "b": round(b, 4), "rho": rho},
        "home_adv_elo": HOME_ADV,
        "elo_k_wc": 60.0,
        "backtest": {"n": n, "accuracy": round(acc, 4), "brier": round(br, 4),
                     "window": "walk-forward, all internationals since 2024-01-01"},
        "hosts": ["Mexico", "United States", "Canada"],
        "groups": groups,
        "elo": elo_out,
        "form": form,
        "goals": goals,
        "h2h": h2h,
        "fixtures": fixtures,
        "bracket": {"r32": r32, "r16": r16, "qf": qf, "sf": sf,
                     "third": third, "final": final},
    }
    with open("site/wc_model.json", "w") as f:
        json.dump(model, f)
    print(f"Wrote site/wc_model.json")

    top = sorted(elo_out.items(), key=lambda kv: -kv[1])[:10]
    print("\nTop 10 by Elo:")
    for t, r in top:
        print(f"  {t:20s} {r:7.1f}")


if __name__ == "__main__":
    main()
