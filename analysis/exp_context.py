"""Experiment: do data-FITTED context features improve forecasting out-of-sample?
Feature under test: rest-day differential (days since each team's previous match).
Protocol: (1) expanding-window time-series CV on 2010-2026 history,
          (2) final held-out test on the 104 WC2026 matches (pre-registered set).
Baseline model: Poisson goals with lambda=exp(a +/- b*elo_diff/400).
Augmented:      adds +/- c*rest_diff term. c fit by MLE on train only.
"""
import csv, math, json
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson as spoisson

# ---------- Elo (same rules as the deployed model) ----------
def k_factor(t):
    t=t.lower()
    if t=="fifa world cup": return 60.0
    if "world cup qualification" in t: return 50.0
    if t in ("uefa euro","copa américa","copa america","african cup of nations","afc asian cup","gold cup"): return 50.0
    if "qualification" in t or "nations league" in t: return 40.0
    if t=="friendly": return 20.0
    return 30.0
def margin_mult(d): return 1.0 if d<=1 else 1.5 if d==2 else (11+d)/8
HOME_ADV=80.0

rows=[]
for r in csv.DictReader(open('../data/intl_results.csv')):
    if r['home_score'] in ('','NA'): continue
    rows.append({'date':r['date'],'home':r['home_team'],'away':r['away_team'],
                 'hg':int(float(r['home_score'])),'ag':int(float(r['away_score'])),
                 'tour':r['tournament'],'neutral':r['neutral']=='TRUE'})
rows.sort(key=lambda x:x['date'])

# add WC2026 results (held-out) from the logger
results={}
for line in open('results.jsonl'):
    j=json.loads(line); results[j['event_id']]=j
meta={}
for line in open('predictions.jsonl'):
    p=json.loads(line); meta[p['event_id']]={'home':p['home'],'away':p['away'],'date':p['kickoff'][:10]}
wc_rows=[]
for eid,rr in results.items():
    m=meta.get(eid)
    if not m: continue
    wc_rows.append({'date':m['date'],'home':m['home'],'away':m['away'],
                    'hg':rr['score'][0],'ag':rr['score'][1],'tour':'FIFA World Cup','neutral':True})
wc_rows.sort(key=lambda x:x['date'])

# ---------- sweep Elo + build features (elo_diff, rest_diff) ----------
def build(all_rows):
    elo={}; last_played={}
    feats=[]
    for m in all_rows:
        rh=elo.get(m['home'],1500.0); ra=elo.get(m['away'],1500.0)
        adv=0.0 if m['neutral'] else HOME_ADV
        x=(rh+adv-ra)/400.0
        # rest days (cap 14; unknown -> 7 neutral)
        def rest(team,d):
            lp=last_played.get(team)
            if lp is None: return 7.0
            from datetime import date
            y1,m1,d1=map(int,lp.split('-')); y2,m2,d2=map(int,d.split('-'))
            dd=(date(y2,m2,d2)-date(y1,m1,d1)).days
            return min(max(dd,0),14)
        rest_diff=rest(m['home'],m['date'])-rest(m['away'],m['date'])
        feats.append({'date':m['date'],'x':x,'rest_diff':rest_diff/14.0,  # scale to ~[-1,1]
                      'hg':m['hg'],'ag':m['ag'],'tour':m['tour']})
        # update elo
        e=1/(1+10**(-x)); s=1.0 if m['hg']>m['ag'] else 0.5 if m['hg']==m['ag'] else 0.0
        delta=k_factor(m['tour'])*margin_mult(abs(m['hg']-m['ag']))*(s-e)
        elo[m['home']]=rh+delta; elo[m['away']]=ra-delta
        last_played[m['home']]=m['date']; last_played[m['away']]=m['date']
    return feats

feats=build(rows + wc_rows)  # continuous sweep so WC elo/rest are correct
# split: history (before WC) vs WC test
hist=[f for f in feats if f['date']<'2026-06-11']
wc  =[f for f in feats if f['date']>='2026-06-11']
hist=[f for f in hist if f['date']>='2010-01-01']   # dense era
print(f"History features (>=2010): {len(hist)}  |  WC2026 held-out: {len(wc)}")

# ---------- Poisson NLL fit ----------
def unpack(p,use_rest): 
    a,b=p[0],p[1]; c=p[2] if use_rest else 0.0; return a,b,c
def nll(p,F,use_rest):
    a,b,c=unpack(p,use_rest)
    x=np.array([f['x'] for f in F]); rd=np.array([f['rest_diff'] for f in F])
    hg=np.array([f['hg'] for f in F]); ag=np.array([f['ag'] for f in F])
    lh=np.exp(a+b*x+c*rd); la=np.exp(a-b*x-c*rd)
    # poisson nll for both goals
    ll=(hg*np.log(lh)-lh)+(ag*np.log(la)-la)
    return -np.sum(ll)
def fit(F,use_rest):
    p0=[0.16,0.75,0.0] if use_rest else [0.16,0.75]
    if not use_rest: 
        r=minimize(lambda p:nll([p[0],p[1]],F,False),p0,method='L-BFGS-B'); return [r.x[0],r.x[1],0.0]
    r=minimize(lambda p:nll(p,F,True),p0,method='L-BFGS-B'); return list(r.x)

# 3-way probs from lambdas (Dixon-Coles rho small; use plain for eval consistency)
def probs(lh,la,G=10):
    xs=spoisson.pmf(np.arange(G+1),lh); ys=spoisson.pmf(np.arange(G+1),la)
    ph=pd=pa=0.0
    for i in range(G+1):
        for j in range(G+1):
            p=xs[i]*ys[j]
            if i>j: ph+=p
            elif i==j: pd+=p
            else: pa+=p
    s=ph+pd+pa; return ph/s,pd/s,pa/s
def brier_logloss(F,par):
    a,b,c=par; B=0.0; L=0.0
    for f in F:
        lh=math.exp(a+b*f['x']+c*f['rest_diff']); la=math.exp(a-b*f['x']-c*f['rest_diff'])
        ph,pd,pa=probs(lh,la)
        o='H' if f['hg']>f['ag'] else 'D' if f['hg']==f['ag'] else 'A'
        oh,od,oa=(o=='H'),(o=='D'),(o=='A')
        B+=(ph-oh)**2+(pd-od)**2+(pa-oa)**2
        L+=-math.log(max(1e-9,{'H':ph,'D':pd,'A':pa}[o]))
    return B/len(F),L/len(F)

# ---------- (1) expanding-window time-series CV on history ----------
print("\n=== (1) Expanding-window CV on 2010-2026 history ===")
hist_sorted=sorted(hist,key=lambda f:f['date'])
folds=[]
# 5 folds: train on first k*, test on next slice
cut_dates=['2016-01-01','2018-01-01','2020-01-01','2022-01-01','2024-01-01']
end_dates=['2018-01-01','2020-01-01','2022-01-01','2024-01-01','2026-06-11']
csum_base=csum_rest=0; lsum_base=lsum_rest=0; nfold=0; cvals=[]
for cut,end in zip(cut_dates,end_dates):
    tr=[f for f in hist_sorted if f['date']<cut]
    te=[f for f in hist_sorted if cut<=f['date']<end]
    if len(te)<50: continue
    pb=fit(tr,False); pr=fit(tr,True)
    bb,lb=brier_logloss(te,pb); br,lr=brier_logloss(te,pr)
    csum_base+=bb;csum_rest+=br;lsum_base+=lb;lsum_rest+=lr;nfold+=1;cvals.append(pr[2])
    print(f"  fold train<{cut} test<{end} (n={len(te):4d}): Brier base {bb:.4f} +rest {br:.4f} | rest coef c={pr[2]:+.4f}")
print(f"  MEAN Brier: base {csum_base/nfold:.4f}  +rest {csum_rest/nfold:.4f}  delta {(csum_rest-csum_base)/nfold:+.4f}")
print(f"  MEAN LogLoss: base {lsum_base/nfold:.4f} +rest {lsum_rest/nfold:.4f} delta {(lsum_rest-lsum_base)/nfold:+.4f}")
print(f"  fitted rest coefficients across folds: {[f'{v:+.3f}' for v in cvals]}")

# ---------- (2) final held-out test on WC2026 ----------
print("\n=== (2) Held-out test: fit on ALL pre-WC history, evaluate on WC2026 ===")
pb=fit(hist,False); pr=fit(hist,True)
bb,lb=brier_logloss(wc,pb); br,lr=brier_logloss(wc,pr)
print(f"  Baseline (Elo only):  Brier {bb:.4f}  LogLoss {lb:.4f}")
print(f"  + fitted rest term:   Brier {br:.4f}  LogLoss {lr:.4f}   (rest coef c={pr[2]:+.4f})")
print(f"  Delta Brier: {br-bb:+.4f}   Delta LogLoss: {lr-lb:+.4f}")

# significance of c on full history: bootstrap CI
print("\n=== Rest coefficient c, full-history fit + bootstrap 95% CI ===")
full=fit(hist,True); print(f"  c (full history) = {full[2]:+.4f}")
rng=np.random.default_rng(0); boot=[]
harr=hist
for _ in range(200):
    samp=[harr[i] for i in rng.integers(0,len(harr),len(harr))]
    boot.append(fit(samp,True)[2])
lo,hi=np.percentile(boot,[2.5,97.5])
print(f"  bootstrap 95% CI: [{lo:+.4f}, {hi:+.4f}]   -> {'excludes 0 (significant)' if lo>0 or hi<0 else 'includes 0 (not significant)'}")

# ---- dump exact numbers for figures + paper ----
import json as _json
_dump = {
  "cv_folds": [{"cut":cut,"end":end,"n":len([f for f in hist_sorted if cut<=f['date']<end]),
                "brier_base":None,"brier_rest":None,"coef":None} for cut,end in zip(cut_dates,end_dates)],
  "rest_coef_full": full[2],
  "rest_ci": [float(lo), float(hi)],
  "rest_boot": [float(x) for x in boot],
  "wc_base_brier": bb, "wc_rest_brier": br,
}
# recompute per-fold to store (cheap; reuse)
_ff=[]
for cut,end in zip(cut_dates,end_dates):
    tr=[f for f in hist_sorted if f['date']<cut]; te=[f for f in hist_sorted if cut<=f['date']<end]
    if len(te)<50: continue
    pbn=fit(tr,False); prn=fit(tr,True)
    bbn,_=brier_logloss(te,pbn); brn,_=brier_logloss(te,prn)
    _ff.append({"cut":cut,"end":end,"n":len(te),"brier_base":bbn,"brier_rest":brn,"coef":prn[2]})
_dump["cv_folds"]=_ff
_json.dump(_dump, open("rest_cv.json","w"))
print("\nDumped rest_cv.json")
