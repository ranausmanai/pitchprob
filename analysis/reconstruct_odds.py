"""Replay the tournament: at each date, run the Monte Carlo with only results
known before that date, to reconstruct how each team's title odds evolved."""
import json, math, random
from collections import defaultdict
random.seed(42)

model = json.load(open('../site/wc_model.json'))
BASE_ELO = dict(model['elo'])
HOSTS = set(model['hosts'])
gm = model['goal_map']; A,B,RHO = gm['a'], gm['b'], gm['rho']
GROUPS = model['groups']
KO_START = '2026-06-28'

# results
results = {}
for line in open('results.jsonl'):
    r = json.loads(line); results[r['event_id']] = r
# need home/away/date per event — get from predictions (has home/away/kickoff) + feed for KO
ev_meta = {}
for line in open('predictions.jsonl'):
    p = json.loads(line)
    ev_meta[p['event_id']] = {'home':p['home'],'away':p['away'],'date':p['kickoff'][:10]}
# feed for any events not in predictions (early KO placeholders resolved)
feed = json.load(open('final_feed.json'))
for e in feed['scoreboard']['events']:
    c = e['competitions'][0]
    h = next(x for x in c['competitors'] if x['homeAway']=='home')
    a = next(x for x in c['competitors'] if x['homeAway']=='away')
    ev_meta[e['id']] = {'home':h['team']['displayName'],'away':a['team']['displayName'],'date':e['date'][:10]}

# assemble result list with winner (for KO real-lock)
res_list = []
for eid, r in results.items():
    m = ev_meta.get(eid)
    if not m: continue
    hs,as_ = r['score']
    if hs>as_: w=m['home']
    elif as_>hs: w=m['away']
    else: w=None  # penalties — we don't have the shootout winner in results.jsonl reliably
    res_list.append({'eid':eid,'home':m['home'],'away':m['away'],'date':m['date'],
                     'hs':hs,'as':as_,'winner':w,'outcome':r['outcome']})

def adv(h,a): return (model['home_adv_elo'] if h in HOSTS else 0)-(model['home_adv_elo'] if a in HOSTS else 0)
def lam(h,a,elo):
    x=(elo[h]+adv(h,a)-elo[a])/400
    return math.exp(A+B*x), math.exp(A-B*x)
def poi(l):
    L=math.exp(-l);k=0;p=1.0
    while True:
        p*=random.random()
        if p<=L: return k
        k+=1
def gof(t):
    for g,ts in GROUPS.items():
        if t in ts: return g

# bracket
BR = model['bracket']
mx_third_slots = []  # (r32_idx, pool)
for i,mm in enumerate(BR['r32']):
    for s in mm['slots']:
        if s['type']=='third': mx_third_slots.append((i, tuple(s['pool'])))

def build_elo(known):
    elo = dict(BASE_ELO)
    for r in known:
        if r['home'] not in elo or r['away'] not in elo: continue
        ad=adv(r['home'],r['away'])
        exp=1/(1+10**(-((elo[r['home']]+ad-elo[r['away']])/400)))
        s=1.0 if r['hs']>r['as'] else 0.5 if r['hs']==r['as'] else 0.0
        d=abs(r['hs']-r['as']); mult=1.0 if d<=1 else 1.5 if d==2 else (11+d)/8
        delta=model['elo_k_wc']*mult*(s-exp)
        elo[r['home']]+=delta; elo[r['away']]-=delta
    return elo

def sim_once(elo, known_group, known_ko):
    # group stage
    stats={t:[0,0,0] for t in elo}
    for g,teams in GROUPS.items():
        pass
    # play group matches: real where known, else sample
    # build set of group fixtures from res_list + remaining scheduled (we approximate: all group pairings come from known + simulate unknown via round-robin? )
    # Simplify: we only have played matches in res_list; for group stage snapshots we use known group results + simulate remaining group games by round-robin within group.
    played_pairs=set()
    for r in known_group:
        A2,Bx=stats[r['home']],stats[r['away']]
        A2[1]+=r['hs']-r['as'];A2[2]+=r['hs'];Bx[1]+=r['as']-r['hs'];Bx[2]+=r['as']
        if r['hs']>r['as']:A2[0]+=3
        elif r['hs']<r['as']:Bx[0]+=3
        else:A2[0]+=1;Bx[0]+=1
        played_pairs.add(frozenset([r['home'],r['away']]))
    # simulate unplayed group games (round robin)
    for g,teams in GROUPS.items():
        for i in range(len(teams)):
            for j in range(i+1,len(teams)):
                a,b=teams[i],teams[j]
                if frozenset([a,b]) in played_pairs: continue
                lh,la=lam(a,b,elo); hs,as_=poi(lh),poi(la)
                A2,Bx=stats[a],stats[b]
                A2[1]+=hs-as_;A2[2]+=hs;Bx[1]+=as_-hs;Bx[2]+=as_
                if hs>as_:A2[0]+=3
                elif hs<as_:Bx[0]+=3
                else:A2[0]+=1;Bx[0]+=1
    key=lambda t:(stats[t][0],stats[t][1],stats[t][2],random.random())
    rank={};thirds=[]
    for g,teams in GROUPS.items():
        o=sorted(teams,key=key,reverse=True)
        rank['1'+g]=o[0];rank['2'+g]=o[1];thirds.append(o[2])
    thirds.sort(key=key,reverse=True); qual=thirds[:8]
    byg={gof(t):t for t in qual}
    # assign thirds (backtracking)
    used=set();assign={}
    order=sorted(range(len(mx_third_slots)),key=lambda i:sum(1 for gg in mx_third_slots[i][1] if gg in byg))
    def bt(i):
        if i==len(order): return True
        idx,pool=mx_third_slots[order[i]]
        for gg in pool:
            if gg in byg and gg not in used:
                used.add(gg);assign[idx]=byg[gg]
                if bt(i+1): return True
                used.discard(gg);assign.pop(idx,None)
        return False
    if not bt(0):
        left=[gg for gg in byg if gg not in used]
        for idx,pool in mx_third_slots:
            if idx not in assign and left: assign[idx]=byg[left.pop()]
    def resolve(slot,w=None):
        t=slot['type']
        if t=='group': return rank[str(slot['rank'])+slot['group']]
        if t=='third': return assign.get(BR['r32'].index(next(m for m in BR['r32'] if slot in m['slots'])))
        return w[slot['idx']] if w else None
    def playKO(a,b):
        k='|'.join(sorted([a,b]))
        if k in known_ko: return known_ko[k]
        lh,la=lam(a,b,elo);hs,as_=poi(lh),poi(la)
        if hs==as_: hs+=poi(lh/3);as_+=poi(la/3)
        if hs==as_: return random.choice([a,b])
        return a if hs>as_ else b
    # r32
    def res_slot(slot, w32=None,w16=None,wqf=None,wsf=None):
        t=slot['type']
        if t=='group': return rank[str(slot['rank'])+slot['group']]
        if t=='third':
            for i,mm in enumerate(BR['r32']):
                if slot in mm['slots']: return assign.get(i)
        if t=='w32': return w32[slot['idx']]
        if t=='w16': return w16[slot['idx']]
        if t=='wqf': return wqf[slot['idx']]
        if t=='wsf': return wsf[slot['idx']]
    w32=[]
    for mm in BR['r32']:
        a=res_slot(mm['slots'][0]); b=res_slot(mm['slots'][1]); w32.append(playKO(a,b))
    w16=[playKO(res_slot(mm['slots'][0],w32),res_slot(mm['slots'][1],w32)) for mm in BR['r16']]
    wqf=[playKO(res_slot(mm['slots'][0],w32,w16),res_slot(mm['slots'][1],w32,w16)) for mm in BR['qf']]
    wsf=[playKO(res_slot(mm['slots'][0],w32,w16,wqf),res_slot(mm['slots'][1],w32,w16,wqf)) for mm in BR['sf']]
    fa=res_slot(BR['final']['slots'][0],w32,w16,wqf,wsf); fb=res_slot(BR['final']['slots'][1],w32,w16,wqf,wsf)
    return playKO(fa,fb)

# snapshot dates
all_dates=sorted(set(r['date'] for r in res_list))
snapshots=['pre']+all_dates
N=2000
series=defaultdict(list); labels=[]
for snap in snapshots:
    if snap=='pre':
        known=[]; label='Pre-tournament'
    else:
        known=[r for r in res_list if r['date']<snap]; label=snap
    elo=build_elo(known)
    kg=[r for r in known if r['date']<KO_START]
    kko={'|'.join(sorted([r['home'],r['away']])):r['winner'] for r in known if r['date']>=KO_START and r['winner']}
    champ=defaultdict(int)
    for _ in range(N):
        champ[sim_once(elo,kg,kko)]+=1
    labels.append(label)
    for t in BASE_ELO: series[t].append(round(champ.get(t,0)/N,4))
# final locked snapshot: Spain 100
labels.append('Champion')
for t in BASE_ELO: series[t].append(1.0 if t=='Spain' else 0.0)

# keep only teams that ever exceeded 4%
keep={t for t,v in series.items() if max(v)>=0.04}
out={'labels':labels,'series':{t:series[t] for t in keep}}
json.dump(out,open('odds_evolution.json','w'))
print("dates:",len(labels),"| teams tracked:",len(keep))
# print spain's arc
sp=series['Spain']
print("Spain arc:", [f"{x:.0%}" for x in sp])
