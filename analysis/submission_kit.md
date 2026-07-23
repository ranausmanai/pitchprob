# PitchProb paper — submission kit

Everything needed to put the paper on arXiv and submit to an applied sports-analytics venue. Copy-paste ready.

---

## A. arXiv submission

**Title**
Nothing Up Our Sleeve: A Pre-Registered Elo–Poisson Forecast of the 2026 World Cup, and a Null Result for Match-Context Adjustments

**Authors**
Rana Usman

**Primary category:** `stat.AP` (Statistics — Applications)
**Cross-list:** `cs.LG` (Machine Learning), optionally `stat.ME`

**Comments field** (paste into arXiv "Comments"):
> 10 pages, 5 figures, 1 table. Pre-registered forecast log, results, and analysis code available at https://github.com/ranausmanai/pitchprob. Live system and report: https://pitchprob.xyz

**License:** CC BY 4.0 (recommended — maximizes citation/reuse)

**Abstract** (condensed to 1,864 chars — under arXiv’s 1,920 limit; paste as-is):

> Public football forecasts are seldom pre-registered: they are typically reported after outcomes are known, are not timestamped, and can be revised or selectively highlighted, which frustrates honest evaluation. We describe a deliberately auditable alternative. For all 104 matches of the 2026 FIFA World Cup, a live system committed a full win/draw/loss probability vector and a most-likely scoreline before each kickoff, timestamped to an append-only, publicly mirrored log (810 snapshots). The forecaster composes an importance-weighted Elo rating (estimated on 49,405 international matches, 1872–2026), a bivariate-Poisson goal model with a Dixon–Coles low-score correction, and a Monte Carlo simulation of the remaining tournament. On the 103 matches with a matched pre-kickoff forecast, three-way accuracy was 63.1% (Wilson 95% CI [0.535, 0.718]). Against a climatological base-rate reference the model showed positive skill on every proper score — Brier skill score +0.195, ranked-probability skill score +0.277, logarithmic skill +0.174 — and a Wilcoxon signed-rank test on per-match Brier scores rejects equality with the reference at p < 0.001. A Murphy decomposition attributes the Brier score to low reliability (0.013) and substantial resolution (0.066); the reliability diagram indicates mild under-confidence, this tournament having produced fewer upsets than the forecasts implied. We then test whether the model's match-context adjustments (heat, rest, travel) carry usable signal by estimating rather than assuming them. A rest-day differential fit on 15,817 historical matches yields a coefficient indistinguishable from zero (bootstrap 95% CI [−0.045, +0.058]) with no out-of-sample improvement in blocked cross-validation or on the held-out World Cup; stadium temperature shows no detectable effect and a sign opposite to the assumed adjustment.

**Note on first-time arXiv submission:** `stat.AP` and `cs.LG` may require an **endorsement** for a first-time author. If prompted, request endorsement from any published author in the category (a former professor, colleague, or co-author), or email arXiv moderation. The paper meets the bar; endorsement is a formality about author identity, not quality.

---

## B. MIT Sloan Sports Analytics Conference — Research Paper track

Sloan's research competition rewards *applied insight and rigor over methodological novelty* — a good fit, provided the pitch leads with the honest angle, not a false claim of a new algorithm.

**One-paragraph summary (for the submission form):**

> Public football predictions are almost never pre-registered — they are reported after the fact, unstamped, and easy to revise or cherry-pick. We deployed a live World Cup 2026 forecasting model that logged every match prediction, timestamped, before kickoff, to a public append-only record (810 snapshots). This lets us evaluate the model with zero hindsight: 63.1% three-way accuracy over 103 matches, calibrated but mildly under-confident on a low-upset tournament, with the eventual finalists and champion among its correct headline calls. We then ask whether the "clever" part — adjusting matches for heat, rest, and travel — actually helps. Learning a rest-day effect from 15,817 historical matches returns a coefficient indistinguishable from zero and no out-of-sample gain; stadium heat shows no detectable effect and even the wrong sign. The contribution is a transparent, reproducible protocol and an honestly reported null, offered as a standard for public sports forecasting.

**Why it fits Sloan (talking points for the cover note):**
- Real-world deployment, not a retrospective simulation.
- A genuinely novel *practice* (pre-registration + public audit trail) even though the model is standard.
- A pre-registered negative result — the field's credibility problem addressed head-on.
- Fully reproducible; open data and code.

**Positioning caveat (be honest in the cover letter):** frame the contribution as *transparency and evaluation practice*, not a new forecasting method. Reviewers respect the honesty and penalize overclaiming.

---

## C. Other viable homes (in order)

1. **arXiv** — do this first regardless; establishes the timestamped record and is citable immediately.
2. **MIT Sloan Sports Analytics Conference** — research paper competition (annual, ~Fall deadline).
3. **StatsBomb Conference** / **Opta Forum** — applied football-analytics audiences.
4. **Journal of Sports Analytics** (IOS Press) or **Journal of Quantitative Analysis in Sports** — if you want peer-reviewed journal placement; both take applied evaluation + negative results.

---

## D. Before submitting — a short checklist

- [x] Repo public at https://github.com/ranausmanai/pitchprob; URL filled in above.
- [ ] Add a `LICENSE` (MIT for code, CC BY 4.0 for the paper/data).
- [x] Five publication figures included (title-odds race, reliability diagram, running Brier, rest-day null, heat scatter). One more human proofreading pass on the PDF is still worthwhile.
- [ ] Optionally register the pre-registration claim's provenance: note the server log's first/last timestamps in an appendix so the "before kickoff" claim is checkable.
- [ ] Consider adding the exact bootstrap seed and fold dates to an appendix for full reproducibility (already in `exp_context.py`).
