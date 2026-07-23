"""Publication figures + scholarly statistics for the PitchProb paper."""
import json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats

plt.rcParams.update({
    'font.family':'serif','font.serif':['Palatino','Palatino Linotype','DejaVu Serif'],
    'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':7.5,
    'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':0.8,
    'axes.edgecolor':'#333','figure.dpi':110,'savefig.dpi':300,'savefig.bbox':'tight',
})
INK='#1a1c22'; GOLD='#b8860b'; GOLD2='#e0a92e'; BLUE='#2f6fb0'; GREY='#9aa3af'; RED='#c0392b'; GREEN='#2e8b57'

# ---------- load ----------
results={}
for l in open('results.jsonl'): r=json.loads(l); results[r['event_id']]=r
preds={}
for l in open('predictions.jsonl'):
    p=json.loads(l); e=p['event_id']
    if e not in preds or p['logged_at']>preds[e]['logged_at']: preds[e]=p
rows=[(preds[e],r) for e,r in results.items() if e in preds]
rows.sort(key=lambda pr: pr[0]['kickoff'])
n=len(rows)

# ---------- scholarly stats ----------
def onehot(o): return {'H':(1,0,0),'D':(0,1,0),'A':(0,0,1)}[o]
def rps(p_vec, o_vec):  # ranked probability score (ordinal H>D>A), lower better
    cp=np.cumsum(p_vec); co=np.cumsum(o_vec)
    return np.sum((cp-co)**2)/(len(p_vec)-1)

# base rates (climatology reference)
base_counts=np.array([sum(1 for _,r in rows if r['outcome']==k) for k in 'HDA'],float)
clim=base_counts/base_counts.sum()

bs_m=bs_ref=rps_m=rps_ref=ll_m=ll_ref=0.0; correct=0
per_match_bs=[]; per_match_bs_ref=[]
cal_p=[]; cal_o=[]
for p,r in rows:
    pv=np.array([p['probs']['H'],p['probs']['D'],p['probs']['A']])
    ov=np.array(onehot(r['outcome']),float)
    bm=np.sum((pv-ov)**2); bref=np.sum((clim-ov)**2)
    bs_m+=bm; bs_ref+=bref; per_match_bs.append(bm); per_match_bs_ref.append(bref)
    rps_m+=rps(pv,ov); rps_ref+=rps(clim,ov)
    ll_m+=-math.log(max(1e-12,pv[{'H':0,'D':1,'A':2}[r['outcome']]]))
    ll_ref+=-math.log(max(1e-12,clim[{'H':0,'D':1,'A':2}[r['outcome']]]))
    correct+= (p['pick']==r['outcome'])
    for k,idx in zip('HDA',range(3)): cal_p.append(pv[idx]); cal_o.append(1.0 if r['outcome']==k else 0.0)
bs_m/=n; bs_ref/=n; rps_m/=n; rps_ref/=n; ll_m/=n; ll_ref/=n
cal_p=np.array(cal_p); cal_o=np.array(cal_o)

def wilson(k,nn,z=1.96):
    if nn==0: return (0,0)
    ph=k/nn; d=1+z*z/nn
    c=(ph+z*z/(2*nn))/d; hw=z*math.sqrt(ph*(1-ph)/nn+z*z/(4*nn*nn))/d
    return c-hw,c+hw
acc=correct/n; acc_lo,acc_hi=wilson(correct,n)
bss=1-bs_m/bs_ref; rpss=1-rps_m/rps_ref; llss=1-ll_m/ll_ref

# Murphy decomposition of Brier (per class averaged), K bins
def murphy(p,o,K=10):
    bins=np.linspace(0,1,K+1); rel=res=0.0; obar=o.mean()
    for i in range(K):
        m=(p>=bins[i])&(p<bins[i+1]) if i<K-1 else (p>=bins[i])&(p<=bins[i+1])
        if m.sum()==0: continue
        pk=p[m].mean(); ok=o[m].mean(); nk=m.sum()
        rel+=nk*(pk-ok)**2; res+=nk*(ok-obar)**2
    return rel/len(p), res/len(p), obar*(1-obar)
rel,reso,unc=murphy(cal_p,cal_o)

print("=== SCHOLARLY STATS (paste into paper) ===")
print(f"n matches: {n}")
print(f"Accuracy: {acc:.3f}  Wilson95 [{acc_lo:.3f},{acc_hi:.3f}]  ({correct}/{n})")
print(f"Brier: model {bs_m:.4f}  ref(clim) {bs_ref:.4f}  BSS {bss:+.3f}")
print(f"RPS:   model {rps_m:.4f}  ref {rps_ref:.4f}  RPSS {rpss:+.3f}")
print(f"LogLoss: model {ll_m:.4f} ref {ll_ref:.4f}  skill {llss:+.3f}")
print(f"Brier decomposition: reliability {rel:.4f} (0=perfect), resolution {reso:.4f}, uncertainty {unc:.4f}")
# paired test model vs climatology per-match Brier
w=stats.wilcoxon(per_match_bs, per_match_bs_ref)
print(f"Wilcoxon model vs climatology per-match Brier: stat {w.statistic:.0f}, p {w.pvalue:.2e}")

# ================= FIGURE 1: championship race =================
odds=json.load(open('odds_evolution.json'))
labels=odds['labels']; N=len(labels)
fig,ax=plt.subplots(figsize=(6.6,3.5))
phases=[(0,18,'Group'),(18,24,'R32'),(24,28,'R16'),(28,32,'QF'),(32,35,'SF'),(35,36,'F')]
for i,(a,b,lab) in enumerate(phases):
    if i%2: ax.axvspan(a,b,color='#000',alpha=0.035,lw=0)
    ax.text((a+b)/2,103,lab,ha='center',va='bottom',fontsize=6.5,color='#888')
xs=np.arange(N)
order=sorted(odds['series'].items(),key=lambda kv:-max(kv[1]))
for name,arc in order:
    y=np.array(arc)*100
    if name=='Spain':
        ax.plot(xs,y,color=GOLD,lw=2.6,zorder=5,solid_capstyle='round')
        ax.text(N-0.5,100,' Spain',color=GOLD,fontsize=8,fontweight='bold',va='center')
    elif name=='Argentina':
        ax.plot(xs,y,color=BLUE,lw=1.7,zorder=4)
        ax.text(35,arc[35]*100,' Argentina',color=BLUE,fontsize=7,va='center')
    else:
        ax.plot(xs,y,color=GREY,lw=0.9,alpha=0.55,zorder=2)
ax.set_ylim(0,108); ax.set_xlim(0,N+2)
ax.set_ylabel('Championship probability (%)'); ax.set_yticks([0,25,50,75,100])
tick_i=[0,9,18,28,36]; ax.set_xticks(tick_i)
ax.set_xticklabels(['Start','Jun 20','Jun 28','Jul 09','Final'])
ax.spines[['top','right']].set_visible(False)
ax.set_title('Modelled title probability across the tournament',loc='left',fontsize=9.5,fontweight='bold')
fig.savefig('fig1_race.png'); plt.close(fig)

# ================= FIGURE 2: reliability diagram =================
fig,(ax,axh)=plt.subplots(2,1,figsize=(4.2,4.6),gridspec_kw={'height_ratios':[3,1],'hspace':0.08},sharex=True)
bins=np.array([0,.1,.2,.3,.4,.5,.6,.75,1.01])
bx=[]; by=[]; berr_lo=[]; berr_hi=[]; bn=[]
for i in range(len(bins)-1):
    m=(cal_p>=bins[i])&(cal_p<bins[i+1])
    if m.sum()<3: continue
    pk=cal_p[m].mean(); ok=cal_o[m].mean(); k=int(cal_o[m].sum()); nn=int(m.sum())
    lo,hi=wilson(k,nn)
    bx.append(pk); by.append(ok); berr_lo.append(ok-lo); berr_hi.append(hi-ok); bn.append(nn)
bx=np.array(bx); by=np.array(by)
ax.plot([0,1],[0,1],'--',color='#999',lw=1,zorder=1,label='perfect calibration')
ax.errorbar(bx,by,yerr=[berr_lo,berr_hi],fmt='o-',color=GOLD,ecolor=GREY,elinewidth=1,
            capsize=2.5,ms=5,lw=1.4,zorder=3,label='PitchProb (95% Wilson CI)')
ax.fill_between([0.4,0.8],[0.4,0.8],[0.68,0.9],color=GREEN,alpha=0.06)
ax.set_ylabel('Observed frequency'); ax.set_xlim(0,0.9); ax.set_ylim(0,0.95)
ax.spines[['top','right']].set_visible(False)
ax.legend(loc='upper left',frameon=False)
ax.set_title('Reliability diagram (all H/D/A forecasts)',loc='left',fontsize=9.5,fontweight='bold')
ax.annotate('under-confident\n(favourites win more\nthan predicted)',xy=(0.63,0.80),xytext=(0.28,0.86),
            fontsize=6.8,color=GREEN,ha='left',arrowprops=dict(arrowstyle='->',color=GREEN,lw=0.8))
axh.hist(cal_p,bins=np.linspace(0,0.9,19),color=GREY,alpha=0.7)
axh.set_ylabel('count'); axh.set_xlabel('Forecast probability'); axh.spines[['top','right']].set_visible(False)
fig.savefig('fig2_reliability.png'); plt.close(fig)

# ================= FIGURE 3: cumulative Brier skill =================
fig,ax=plt.subplots(figsize=(6.4,3.0))
cum_m=np.cumsum(per_match_bs)/np.arange(1,n+1)
cum_r=np.cumsum(per_match_bs_ref)/np.arange(1,n+1)
xi=np.arange(1,n+1)
ax.plot(xi,cum_m,color=GOLD,lw=2,label='PitchProb',zorder=3)
ax.plot(xi,cum_r,color=GREY,lw=1.4,ls='--',label='Climatology (base rates)',zorder=2)
for x,lab in [(69,'knockouts')]:
    ax.axvline(x,color='#bbb',lw=0.8,ls=':'); ax.text(x+1,0.70,lab,fontsize=6.5,color='#999',rotation=90,va='top')
ax.set_xlabel('Match (chronological)'); ax.set_ylabel('Cumulative mean Brier score')
ax.set_ylim(0.40,0.72); ax.set_xlim(1,n)
ax.spines[['top','right']].set_visible(False); ax.legend(frameon=False,loc='upper right')
ax.set_title('Running Brier score vs. a base-rate baseline',loc='left',fontsize=9.5,fontweight='bold')
fig.savefig('fig3_cumbrier.png'); plt.close(fig)

# ================= FIGURE 4: rest-day coefficient null =================
rc=json.load(open('rest_cv.json'))
fig,ax=plt.subplots(figsize=(5.6,2.9))
coefs=[f['coef'] for f in rc['cv_folds']]; flabs=[f["end"][:4] for f in rc['cv_folds']]
ypos=np.arange(len(coefs))
ax.axvspan(rc['rest_ci'][0],rc['rest_ci'][1],color=GOLD,alpha=0.12,label='full-sample 95% CI')
ax.axvline(0,color=RED,lw=1,ls='--',zorder=1,label='no effect')
ax.axvline(rc['rest_coef_full'],color=GOLD,lw=1.4,zorder=2,label=f"full estimate {rc['rest_coef_full']:+.3f}")
ax.scatter(coefs,ypos,color=BLUE,s=34,zorder=4,label='per-fold estimate')
ax.set_yticks(ypos); ax.set_yticklabels([f'fold ending {l}' for l in flabs])
ax.set_xlabel('Fitted rest-day coefficient $c$'); ax.set_xlim(-0.09,0.09)
ax.spines[['top','right']].set_visible(False); ax.legend(frameon=False,fontsize=6.8,loc='lower right')
ax.set_title('Rest-day effect fit from 15,817 matches — indistinguishable from zero',loc='left',fontsize=9,fontweight='bold')
fig.savefig('fig4_rest_null.png'); plt.close(fig)

# ================= FIGURE 5: heat vs goals =================
temps=[]; goals=[]
for p,r in rows:
    t=p['context'].get('temp_c')
    if t is None: continue
    temps.append(t); goals.append(r['score'][0]+r['score'][1])
temps=np.array(temps,float); goals=np.array(goals,float)
sl,ic,rr,pp,se=stats.linregress(temps,goals)
fig,ax=plt.subplots(figsize=(5.2,3.0))
jit=goals+np.random.default_rng(1).normal(0,0.06,len(goals))
ax.scatter(temps,jit,s=16,color=BLUE,alpha=0.45,edgecolor='none')
xx=np.linspace(temps.min(),temps.max(),50)
ax.plot(xx,ic+sl*xx,color=GOLD,lw=2,zorder=3,label=f'OLS slope {sl:+.3f} goals/°C (p={pp:.2f})')
# CI band
resid=goals-(ic+sl*temps); s_err=np.sqrt(np.sum(resid**2)/(len(temps)-2))
sxx=np.sum((temps-temps.mean())**2)
ci=1.96*s_err*np.sqrt(1/len(temps)+(xx-temps.mean())**2/sxx)
ax.fill_between(xx,ic+sl*xx-ci,ic+sl*xx+ci,color=GOLD,alpha=0.12)
ax.set_xlabel('Apparent temperature at kickoff (°C)'); ax.set_ylabel('Total goals in match')
ax.spines[['top','right']].set_visible(False); ax.legend(frameon=False,loc='upper left',fontsize=7)
ax.set_title('Stadium heat vs. scoring — no significant effect',loc='left',fontsize=9.5,fontweight='bold')
fig.savefig('fig5_heat.png'); plt.close(fig)

print("\nHeat regression: slope %.4f goals/C, p=%.3f, r=%.3f"%(sl,pp,rr))
print("Figures written: fig1_race, fig2_reliability, fig3_cumbrier, fig4_rest_null, fig5_heat (PNG 300dpi)")
