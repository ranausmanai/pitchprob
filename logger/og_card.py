#!/usr/bin/env python3
"""Generate PitchProb's share card (Open Graph / Twitter image) + PNG favicons.

The OG card is dynamic: it shows the live match (with score) if one is in play,
otherwise the next upcoming match with the model's win probabilities. Runs from
cron every 3 minutes on the VPS so the card people share is always current.

Outputs into /var/www/pitchprob/ :
  og.png            1200x630 share card
  favicon-32.png, favicon-180.png, favicon-512.png

Usage: og_card.py [SITE_DIR]   (defaults to /var/www/pitchprob)
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

from PIL import Image, ImageDraw, ImageFont

SITE = sys.argv[1] if len(sys.argv) > 1 else "/var/www/pitchprob"
DATA = os.path.join(SITE, "data")

BG = (7, 9, 15)
PANEL = (22, 27, 36)
TEXT = (238, 243, 248)
MUTED = (147, 160, 180)
HOME = (65, 196, 99)
DRAW = (217, 166, 46)
AWAY = (90, 167, 255)
GOLD = (232, 187, 77)
LIVE = (255, 92, 82)

FONTDIR = "/usr/share/fonts/truetype/dejavu"
if not os.path.isdir(FONTDIR):  # macOS dev fallback
    FONTDIR = "/System/Library/Fonts/Supplemental"


def font(size, bold=True):
    for name in (("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
                 ("Arial Bold.ttf" if bold else "Arial.ttf")):
        p = os.path.join(FONTDIR, name)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def load(path, default):
    try:
        return json.load(open(path))
    except Exception:
        return default


def pmf(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def dc_tau(x, y, lh, la, rho):
    if x == 0 and y == 0: return 1 - lh * la * rho
    if x == 0 and y == 1: return 1 + lh * rho
    if x == 1 and y == 0: return 1 + la * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0


def probs(lh, la, rho, G=9):
    ph = pd = pa = 0.0
    for x in range(G + 1):
        for y in range(G + 1):
            p = pmf(x, lh) * pmf(y, la) * dc_tau(x, y, lh, la, rho)
            if x > y: ph += p
            elif x == y: pd += p
            else: pa += p
    s = ph + pd + pa
    return ph / s, pd / s, pa / s


def pick_match(model):
    """Return (label, home, away, dict) for the card's headline match."""
    feed = load(os.path.join(DATA, "feed.json"), {})
    sb = feed.get("scoreboard", {})
    hosts = set(model["hosts"])
    elo = dict(model["elo"])
    evs = []
    for e in sb.get("events", []):
        c = e["competitions"][0]
        h = next(x for x in c["competitors"] if x["homeAway"] == "home")
        a = next(x for x in c["competitors"] if x["homeAway"] == "away")
        if any(k in h["team"]["displayName"] for k in ("Winner", "Place", "Third")):
            continue
        evs.append({"date": e["date"], "home": h["team"]["displayName"],
                    "away": a["team"]["displayName"],
                    "hs": int(h.get("score") or 0), "as": int(a.get("score") or 0),
                    "state": e["status"]["type"]["state"],
                    "completed": e["status"]["type"]["completed"],
                    "detail": e["status"]["type"]["shortDetail"]})
    evs.sort(key=lambda x: x["date"])
    # live Elo replay
    for ev in evs:
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

    gm = model["goal_map"]

    def pr(ev):
        adv = (model["home_adv_elo"] if ev["home"] in hosts else 0) \
            - (model["home_adv_elo"] if ev["away"] in hosts else 0)
        x = (elo[ev["home"]] + adv - elo[ev["away"]]) / 400
        return probs(math.exp(gm["a"] + gm["b"] * x), math.exp(gm["a"] - gm["b"] * x), gm["rho"])

    live = [e for e in evs if e["state"] == "in" and e["home"] in elo]
    if live:
        ev = live[0]
        return ("LIVE", ev, pr(ev))
    nxt = [e for e in evs if e["state"] == "pre" and e["home"] in elo]
    if nxt:
        ev = nxt[0]
        return ("NEXT MATCH", ev, pr(ev))
    done = [e for e in evs if e["completed"] and e["home"] in elo]
    if done:
        ev = done[-1]
        return ("LATEST RESULT", ev, pr(ev))
    return (None, None, None)


def rounded(draw, box, r, fill):
    draw.rounded_rectangle(box, radius=r, fill=fill)


def center(draw, cx, y, text, fnt, fill):
    w = draw.textlength(text, font=fnt)
    draw.text((cx - w / 2, y), text, font=fnt, fill=fill)


def make_og(model):
    label, ev, pr = pick_match(model)
    W, H = 1200, 630
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # subtle top glow
    for i in range(160):
        a = int(22 * (1 - i / 160))
        d.line([(0, i), (W, i)], fill=(7 + a // 3, 9 + a, 15 + a // 3))

    center(d, W / 2, 54, "PITCHPROB", font(58), GOLD)
    center(d, W / 2, 124, "World Cup 2026  ·  Live Prediction Engine", font(28, False), MUTED)

    if not ev:
        center(d, W / 2, 300, "Who will win the World Cup?", font(46), TEXT)
        center(d, W / 2, 370, "Live odds from 10,000 simulations, updated every minute",
               font(26, False), MUTED)
    else:
        # status pill
        pill_txt = f"● {ev['detail']}" if label == "LIVE" else label
        pw = d.textlength(pill_txt, font=font(24)) + 40
        pill_col = LIVE if label == "LIVE" else GOLD
        rounded(d, [W / 2 - pw / 2, 168, W / 2 + pw / 2, 212], 22, PANEL)
        center(d, W / 2, 178, pill_txt, font(24), pill_col)

        # teams + score/vs
        ty = 250
        home, away = ev["home"], ev["away"]
        d.text((80, ty), home, font=font(46), fill=TEXT)
        aw = d.textlength(away, font=font(46))
        d.text((W - 80 - aw, ty), away, font=font(46), fill=TEXT)
        mid = "vs"
        if label in ("LIVE", "LATEST RESULT"):
            mid = f"{ev['hs']} – {ev['as']}"
        center(d, W / 2, ty - 4, mid, font(56), TEXT if label != "LIVE" else LIVE)

        # probability bar
        ph, pdr, pa = pr
        bx0, bx1, by0, by1 = 80, W - 80, 360, 410
        bw = bx1 - bx0
        wh = int(bw * ph); wd = int(bw * pdr)
        rounded(d, [bx0, by0, bx1, by1], 14, PANEL)
        seg = Image.new("RGB", (bw, by1 - by0), PANEL)
        sd = ImageDraw.Draw(seg)
        sd.rectangle([0, 0, wh, by1 - by0], fill=HOME)
        sd.rectangle([wh, 0, wh + wd, by1 - by0], fill=DRAW)
        sd.rectangle([wh + wd, 0, bw, by1 - by0], fill=AWAY)
        mask = Image.new("L", (bw, by1 - by0), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, bw, by1 - by0], radius=14, fill=255)
        img.paste(seg, (bx0, by0), mask)
        d = ImageDraw.Draw(img)

        def lab(x, txt, col):
            d.text((x, by1 + 14), txt, font=font(26), fill=col)
        lab(bx0, f"{home}  {round(ph*100)}%", HOME)
        dtxt = f"Draw {round(pdr*100)}%"
        center(d, W / 2, by1 + 14, dtxt, font(26), DRAW)
        rt = f"{round(pa*100)}%  {away}"
        d.text((bx1 - d.textlength(rt, font=font(26)), by1 + 14), rt, font=font(26), fill=AWAY)

        if label == "NEXT MATCH":
            fav = home if ph > pa and ph > pdr else away if pa > ph and pa > pdr else "Draw"
            center(d, W / 2, 470, f"Model pick: {fav}  ({round(max(ph, pdr, pa)*100)}%)",
                   font(28), TEXT)

    center(d, W / 2, 575, "pitchprob.xyz", font(30), GOLD)
    img.save(os.path.join(SITE, "og.png"), "PNG")

    # favicons from the SVG-equivalent (drawn here so we don't need a rasteriser)
    make_favicons()


def make_favicons():
    for size in (32, 180, 512):
        s = size * 4
        ic = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        d = ImageDraw.Draw(ic)
        d.rounded_rectangle([0, 0, s, s], radius=int(s * 0.22), fill=(15, 36, 25))
        cx = cy = s / 2
        r = s * 0.31
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(44, 94, 58), width=max(2, int(s * 0.035)))
        d.line([(cx, cy - r), (cx, cy + r)], fill=(44, 94, 58), width=max(2, int(s * 0.035)))
        # gold probability arc ~62%
        d.arc([cx - r, cy - r, cx + r, cy + r], start=-90, end=-90 + 223,
              fill=GOLD, width=max(3, int(s * 0.066)))
        # ball
        br = s * 0.135
        d.ellipse([cx - br, cy - br, cx + br, cy + br], fill=(242, 246, 250))
        pr = br * 0.62
        pts = [(cx + pr * math.cos(math.radians(a - 90)),
                cy + pr * math.sin(math.radians(a - 90))) for a in (0, 72, 144, 216, 288)]
        d.polygon(pts, fill=(12, 31, 20))
        ic = ic.resize((size, size), Image.LANCZOS)
        ic.save(os.path.join(SITE, f"favicon-{size}.png"), "PNG")


if __name__ == "__main__":
    model = json.load(open(os.path.join(SITE, "wc_model.json")))
    make_og(model)
    print(datetime.now(timezone.utc).isoformat(timespec="seconds"), "og.png + favicons written")
