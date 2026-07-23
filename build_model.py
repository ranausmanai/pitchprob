"""Build football match prediction model from football-data.co.uk CSVs.

Reads data/<LEAGUE>_<SEASON>.csv files, fits per-league models:
  - Elo ratings (with margin-of-victory multiplier and season regression)
  - Time-decayed Poisson attack/defense strengths (Dixon-Coles style)
Backtests on the final season, then exports site/model.json.
"""

import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, date

DATA_DIR = "data"
OUT_PATH = "site/model.json"

LEAGUES = {
    "E0": "Premier League (England)",
    "SP1": "La Liga (Spain)",
    "D1": "Bundesliga (Germany)",
    "I1": "Serie A (Italy)",
    "F1": "Ligue 1 (France)",
}

# --- Model parameters ---
ELO_START = 1500.0
ELO_K = 24.0
ELO_HOME_ADV = 60.0          # Elo points of home advantage
SEASON_REGRESS = 0.25        # regress 25% toward league mean between seasons
POISSON_HALF_LIFE_DAYS = 390 # weight halves every ~13 months
DC_RHO = -0.10               # Dixon-Coles low-score correlation
MAX_GOALS = 8                # scoreline grid size


def parse_date(s):
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"bad date: {s}")


def load_matches():
    """Return {league: [match dicts sorted by date]}."""
    by_league = defaultdict(list)
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".csv"):
            continue
        league, season = fname[:-4].split("_")
        with open(os.path.join(DATA_DIR, fname), encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if not row.get("HomeTeam") or not row.get("FTHG"):
                    continue
                try:
                    m = {
                        "date": parse_date(row["Date"]),
                        "season": season,
                        "home": row["HomeTeam"].strip(),
                        "away": row["AwayTeam"].strip(),
                        "hg": int(row["FTHG"]),
                        "ag": int(row["FTAG"]),
                    }
                except (ValueError, KeyError):
                    continue
                # bookmaker average odds, if present (for backtest comparison)
                try:
                    m["odds"] = (float(row["AvgH"]), float(row["AvgD"]), float(row["AvgA"]))
                except (ValueError, KeyError, TypeError):
                    m["odds"] = None
                by_league[league].append(m)
    for league in by_league:
        by_league[league].sort(key=lambda m: m["date"])
    return by_league


# ---------------- Elo ----------------

def elo_expected(r_home, r_away):
    return 1.0 / (1.0 + 10 ** (-(r_home + ELO_HOME_ADV - r_away) / 400.0))


def run_elo(matches):
    """Sequentially update Elo; returns final ratings dict."""
    ratings = {}
    current_season = None
    for m in matches:
        if m["season"] != current_season:
            if current_season is not None:
                mean = sum(ratings.values()) / len(ratings)
                for t in ratings:
                    ratings[t] += SEASON_REGRESS * (mean - ratings[t])
            current_season = m["season"]
        rh = ratings.setdefault(m["home"], ELO_START)
        ra = ratings.setdefault(m["away"], ELO_START)
        exp_h = elo_expected(rh, ra)
        score_h = 1.0 if m["hg"] > m["ag"] else (0.5 if m["hg"] == m["ag"] else 0.0)
        margin = abs(m["hg"] - m["ag"])
        mov = math.log(max(margin, 1) + 1.0) * (2.2 / ((rh - ra) * 0.001 * (1 if score_h == 1 else -1) + 2.2))
        delta = ELO_K * mov * (score_h - exp_h)
        ratings[m["home"]] = rh + delta
        ratings[m["away"]] = ra - delta
    return ratings


# ---------------- Poisson attack/defense ----------------

def fit_poisson(matches, ref_date, iters=20):
    """Time-weighted attack/defense strengths via iterative scaling.

    Model: lambda_home = home_adv * base * att[home] * dfn[away]
           lambda_away = base * att[away] * dfn[home]
    """
    weights, teams = [], set()
    for m in matches:
        age = (ref_date - m["date"]).days
        weights.append(0.5 ** (age / POISSON_HALF_LIFE_DAYS))
        teams.update((m["home"], m["away"]))
    teams = sorted(teams)
    att = {t: 1.0 for t in teams}
    dfn = {t: 1.0 for t in teams}

    total_w = sum(weights)
    total_goals = sum((m["hg"] + m["ag"]) * w for m, w in zip(matches, weights))
    base = total_goals / (2.0 * total_w)  # avg goals per team per match
    home_goals = sum(m["hg"] * w for m, w in zip(matches, weights))
    away_goals = sum(m["ag"] * w for m, w in zip(matches, weights))
    home_adv = home_goals / away_goals if away_goals > 0 else 1.2

    for _ in range(iters):
        att_num = defaultdict(float); att_den = defaultdict(float)
        dfn_num = defaultdict(float); dfn_den = defaultdict(float)
        for m, w in zip(matches, weights):
            h, a = m["home"], m["away"]
            att_num[h] += w * m["hg"]
            att_den[h] += w * home_adv * base * dfn[a]
            att_num[a] += w * m["ag"]
            att_den[a] += w * base * dfn[h]
            dfn_num[a] += w * m["hg"]
            dfn_den[a] += w * home_adv * base * att[h]
            dfn_num[h] += w * m["ag"]
            dfn_den[h] += w * base * att[a]
        for t in teams:
            if att_den[t] > 0:
                att[t] = att_num[t] / att_den[t]
            if dfn_den[t] > 0:
                dfn[t] = dfn_num[t] / dfn_den[t]
        # normalize so mean strength is 1
        ma = sum(att.values()) / len(teams)
        md = sum(dfn.values()) / len(teams)
        for t in teams:
            att[t] /= ma
            dfn[t] /= md
        base *= ma * md
    return {"base": base, "home_adv": home_adv, "att": att, "dfn": dfn}


def dc_tau(x, y, lh, la, rho):
    if x == 0 and y == 0:
        return 1 - lh * la * rho
    if x == 0 and y == 1:
        return 1 + lh * rho
    if x == 1 and y == 0:
        return 1 + la * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def poisson_pmf(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def match_probs(lh, la, rho=DC_RHO):
    """Return (p_home, p_draw, p_away) from Dixon-Coles Poisson grid."""
    ph = pd = pa = 0.0
    for x in range(MAX_GOALS + 1):
        px = poisson_pmf(x, lh)
        for y in range(MAX_GOALS + 1):
            p = px * poisson_pmf(y, la) * dc_tau(x, y, lh, la, rho)
            if x > y:
                ph += p
            elif x == y:
                pd += p
            else:
                pa += p
    s = ph + pd + pa
    return ph / s, pd / s, pa / s


def predict_lambdas(pois, elo, home, away):
    """Expected goals for a fixture, blending Poisson strengths with Elo."""
    lh = pois["home_adv"] * pois["base"] * pois["att"][home] * pois["dfn"][away]
    la = pois["base"] * pois["att"][away] * pois["dfn"][home]
    # Elo nudge: shift goal expectation by rating gap
    gap = (elo[home] + ELO_HOME_ADV - elo[away]) / 400.0
    elo_factor = 10 ** (gap * 0.12)
    lh *= math.sqrt(elo_factor)
    la /= math.sqrt(elo_factor)
    return lh, la


# ---------------- Backtest ----------------

def backtest(by_league):
    """Train on all seasons except the last, predict the last season."""
    results = {}
    for league, matches in by_league.items():
        seasons = sorted({m["season"] for m in matches})
        test_season = seasons[-1]
        train = [m for m in matches if m["season"] != test_season]
        test = [m for m in matches if m["season"] == test_season]

        n = correct = 0
        brier = 0.0
        book_correct = 0
        book_n = 0
        # walk forward: refit after each month would be slow; refit per match window
        history = list(train)
        elo = run_elo(history)
        pois = fit_poisson(history, history[-1]["date"])
        last_refit = 0
        for i, m in enumerate(test):
            if i - last_refit >= 50:  # refit every ~50 matches
                elo = run_elo(history)
                pois = fit_poisson(history, history[-1]["date"])
                last_refit = i
            h, a = m["home"], m["away"]
            if h not in pois["att"] or a not in pois["att"]:
                history.append(m)
                continue
            lh, la = predict_lambdas(pois, elo, h, a)
            ph, pd, pa = match_probs(lh, la)
            pick = max(((ph, "H"), (pd, "D"), (pa, "A")))[1]
            actual = "H" if m["hg"] > m["ag"] else ("D" if m["hg"] == m["ag"] else "A")
            n += 1
            correct += pick == actual
            oh = 1.0 if actual == "H" else 0.0
            od = 1.0 if actual == "D" else 0.0
            oa = 1.0 if actual == "A" else 0.0
            brier += (ph - oh) ** 2 + (pd - od) ** 2 + (pa - oa) ** 2
            if m["odds"]:
                inv = [1 / o for o in m["odds"]]
                book_pick = "HDA"[inv.index(max(inv))]
                book_correct += book_pick == actual
                book_n += 1
            history.append(m)
        results[league] = {
            "n": n,
            "accuracy": correct / n if n else 0,
            "brier": brier / n if n else 0,
            "bookmaker_accuracy": book_correct / book_n if book_n else None,
        }
    return results


# ---------------- Export ----------------

def recent_form(matches, team, k=5):
    res = []
    for m in reversed(matches):
        if m["home"] == team:
            res.append("W" if m["hg"] > m["ag"] else ("D" if m["hg"] == m["ag"] else "L"))
        elif m["away"] == team:
            res.append("W" if m["ag"] > m["hg"] else ("D" if m["hg"] == m["ag"] else "L"))
        if len(res) == k:
            break
    return res


def main():
    by_league = load_matches()
    print("Loaded matches:", {k: len(v) for k, v in by_league.items()})

    print("\nBacktesting on final season (train = prior seasons)...")
    bt = backtest(by_league)
    for league, r in bt.items():
        book = f"{r['bookmaker_accuracy']:.1%}" if r["bookmaker_accuracy"] else "n/a"
        print(f"  {league}: accuracy {r['accuracy']:.1%} on {r['n']} matches "
              f"(bookmakers: {book}), brier {r['brier']:.3f}")

    model = {
        "generated": date.today().isoformat(),
        "source": "football-data.co.uk",
        "params": {"elo_home_adv": ELO_HOME_ADV, "dc_rho": DC_RHO,
                   "elo_blend_exp": 0.12, "max_goals": MAX_GOALS},
        "backtest": bt,
        "leagues": {},
    }
    for league, matches in by_league.items():
        elo = run_elo(matches)
        pois = fit_poisson(matches, matches[-1]["date"])
        last_season = sorted({m["season"] for m in matches})[-1]
        current_teams = sorted({m["home"] for m in matches if m["season"] == last_season})
        # head-to-head history (last 6 meetings per pair, current teams only)
        h2h = defaultdict(list)
        for m in matches:
            if m["home"] in current_teams and m["away"] in current_teams:
                key = "|".join(sorted([m["home"], m["away"]]))
                h2h[key].append([m["date"].isoformat(), m["home"], m["hg"], m["ag"], m["away"]])
        h2h = {k: v[-6:] for k, v in h2h.items()}

        model["leagues"][league] = {
            "name": LEAGUES.get(league, league),
            "base": pois["base"],
            "home_adv": pois["home_adv"],
            "teams": {
                t: {
                    "elo": round(elo.get(t, ELO_START), 1),
                    "att": round(pois["att"][t], 4),
                    "dfn": round(pois["dfn"][t], 4),
                    "form": recent_form(matches, t),
                }
                for t in current_teams
            },
            "h2h": h2h,
        }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(model, f)
    size = os.path.getsize(OUT_PATH) / 1024
    print(f"\nWrote {OUT_PATH} ({size:.0f} KB)")


if __name__ == "__main__":
    main()
