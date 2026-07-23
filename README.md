# PitchProb — a pre-registered World Cup 2026 forecast

A probabilistic forecasting model that logged a prediction for **every match of the 2026 FIFA World Cup before kickoff**, to a public, timestamped, append-only record — then evaluated itself honestly against that frozen log once the tournament was over.

📄 **Paper:** [`analysis/pitchprob_paper.pdf`](analysis/pitchprob_paper.pdf) · 🌐 **Live site & report:** [pitchprob.xyz](https://pitchprob.xyz)

---

## What this is

Public football predictions are almost never pre-registered: they are reported after the fact, unstamped, and easy to revise or cherry-pick. This project holds one model to a stricter standard. A live system committed a full win/draw/loss probability vector and a most-likely scoreline for each match *before kickoff*, timestamped to a server-side log (810 snapshots), and mirrored those predictions publicly during the tournament. Every performance claim in the paper is checkable against a record that provably predates the results.

The forecaster is a deliberately standard composition — importance-weighted **Elo** ratings over 49,405 international matches (1872–2026), a **bivariate-Poisson** goal model with the Dixon–Coles correction, and a **Monte Carlo** simulation of the remaining bracket. The contribution is not the method but the protocol: *commit, log publicly, and report the nulls.*

## Headline results (103 matched pre-kickoff forecasts)

| Metric | Value |
|---|---|
| Three-way accuracy | 63.1% (Wilson 95% CI 53.5–71.8%) |
| Brier skill vs. base rates | +0.195 |
| Ranked-probability skill | +0.277 |
| Wilcoxon vs. baseline (per-match Brier) | p < 0.001 |

The model's pre-tournament #1 and #2 seeds finished first and second; its single most-likely final, identified weeks early, was the one that occurred. A direct test of its match-context adjustments (heat, rest, travel) — *fitting* a rest-day effect on 15,817 historical matches rather than assuming it — returned a **robust null** (coefficient 95% CI [−0.045, +0.058], no out-of-sample gain). That negative result is reported in full; the public log made it impossible to omit.

## Repository layout

```
analysis/          The paper and its reproducible analysis
  paper.md                 source
  pitchprob_paper.pdf      typeset preprint (10 pp, 5 figures)
  predictions.jsonl        ★ the pre-registered forecast log (810 snapshots)
  results.jsonl            ★ final results for all 104 matches
  exp_context.py           fitted rest-day experiment (CV + held-out + bootstrap)
  make_figures.py          all statistics + the 5 publication figures
  reconstruct_odds.py      replays the tournament to rebuild title-odds over time
  fig1..fig5 .png          figures
  submission_kit.md        arXiv / Sloan submission materials
build_wc_model.py   builds the model (Elo + goal map + bracket) → site/wc_model.json
build_model.py      club-league variant
data/               historical international results (martj42/international_results)
site/               the live web app (index = report, engine.html = predictor)
logger/             server-side jobs that produced the pre-registered log
```

★ = the core scientific artifacts.

## Reproduce the analysis

```sh
pip install numpy scipy scikit-learn matplotlib pandas
cd analysis
python3 exp_context.py     # rest-day fit: CV folds, held-out WC test, bootstrap CI
python3 make_figures.py    # prints all reported statistics; writes fig1..fig5
```

Both scripts read only `predictions.jsonl`, `results.jsonl`, and the historical CSV, and print the exact numbers cited in the paper. Fold dates and the bootstrap seed are in `exp_context.py`.

## Rebuild the model from scratch

```sh
# refresh historical results, then:
python3 build_wc_model.py     # → site/wc_model.json (Elo, goal map, bracket, groups)
```

## Data sources

- **Historical results:** [`martj42/international_results`](https://github.com/martj42/international_results) — 49,405 international matches, 1872–2026.
- **Live results & schedule:** ESPN public scoreboard API.
- **Weather:** Open-Meteo API.

## Citation

> Usman, R. (2026). *Predicting the 2026 World Cup in Public: A Pre-Registered Evaluation of an Elo–Poisson Forecasting Model and a Null Result for Match-Context Adjustments.* Preprint.

## License

Code: MIT. Paper and the pre-registered forecast dataset: CC BY 4.0. See [`LICENSE`](LICENSE).

Built by [Rana Usman](https://www.linkedin.com/in/ranausmans/).
