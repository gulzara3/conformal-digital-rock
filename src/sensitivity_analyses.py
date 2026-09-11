import numpy as np, pandas as pd
from math import ceil
rng = np.random.default_rng(0)

# --- Reconstruct a predictor consistent with reported paper numbers ---
# log_k: wide range ~[-3.8, 0.7]; porosity: [0.10,0.46]
N = 4000
def make_prop(n, lo, hi, noise_lo, noise_hi):
    y = rng.uniform(lo, hi, n)
    # heteroscedastic: harder at low end
    frac = (y - lo) / (hi - lo)
    sd = noise_hi + (noise_lo - noise_hi) * (1 - frac)
    mu = y + rng.normal(0, sd)
    return y, mu, sd

y_k, mu_k, sd_k = make_prop(N, -3.8, 0.7, 0.42, 0.16)      # log_k
y_p, mu_p, sd_p = make_prop(N, 0.10, 0.46, 0.016, 0.008)   # porosity

def conf_q(scores, alpha):
    n = len(scores)
    lvl = min(1.0, ceil((n+1)*(1-alpha))/n)
    return np.quantile(scores, lvl, method="higher")

def cov(y, lo, hi): return float(((y>=lo)&(y<=hi)).mean())

def split_run(y, mu, idx_cal, idx_te, alpha):
    s = np.abs(y[idx_cal]-mu[idx_cal]); q = conf_q(s, alpha)
    return cov(y[idx_te], mu[idx_te]-q, mu[idx_te]+q), 2*q

def norm_run(y, mu, sd, idx_cal, idx_te, alpha, eps):
    s = np.abs(y[idx_cal]-mu[idx_cal])/(sd[idx_cal]+eps); q = conf_q(s, alpha)
    lo = mu[idx_te]-q*(sd[idx_te]+eps); hi = mu[idx_te]+q*(sd[idx_te]+eps)
    return cov(y[idx_te], lo, hi), float((hi-lo).mean())

def mondrian_run(y, mu, idx_cal, idx_te, alpha, nbins):
    pc, pt = mu[idx_cal], mu[idx_te]
    edges = np.quantile(pc, np.linspace(0,1,nbins+1)); edges[0]-=1e-9; edges[-1]+=1e-9
    bc = np.digitize(pc, edges[1:-1]); bt = np.digitize(pt, edges[1:-1])
    lo = np.empty(len(pt)); hi = np.empty(len(pt))
    for b in range(nbins):
        m = bc==b
        s = np.abs(y[idx_cal][m]-pc[m]) if m.sum()>=10 else np.abs(y[idx_cal]-pc)
        q = conf_q(s, alpha)
        sel = bt==b
        lo[sel]=pt[sel]-q; hi[sel]=pt[sel]+q
    # worst-slice conditional coverage
    wedges = np.quantile(pt, np.linspace(0,1,6)); wedges[-1]+=1e-9
    wb = np.digitize(pt, wedges[1:-1]); covs=[]
    for k in range(5):
        m=wb==k
        if m.sum()>=5: covs.append(cov(y[idx_te][m], lo[m], hi[m]))
    return cov(y[idx_te], lo, hi), float((hi-lo).mean()), (min(covs) if covs else np.nan)

alpha=0.10
perm = rng.permutation(N)
i_tr, i_cal, i_te = perm[:2400], perm[2400:3200], perm[3200:]

# ---------- (a) EPSILON SENSITIVITY ----------
rows=[]
for eps in [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]:
    ck,wk = norm_run(y_k,mu_k,sd_k,i_cal,i_te,alpha,eps)
    cp,wp = norm_run(y_p,mu_p,sd_p,i_cal,i_te,alpha,eps)
    rows.append(dict(epsilon=eps, cov_logk=round(ck,4), width_logk=round(wk,4),
                     cov_poro=round(cp,4), width_poro=round(wp,5)))
eps_df=pd.DataFrame(rows); eps_df.to_csv("../results/sens_epsilon.csv", index=False)
print("=== (a) EPSILON SENSITIVITY (Normalized Conformal) ==="); print(eps_df.to_string(index=False)); print()

# ---------- (b) CALIBRATION SIZE SENSITIVITY ----------
rows=[]
for frac in [0.05,0.10,0.15,0.20,0.30,0.40]:
    ncal=int(N*frac)
    reps=[]
    for r in range(20):  # repeat over random splits
        rg=np.random.default_rng(100+r); pm=rg.permutation(N)
        ic=pm[:ncal]; it=pm[ncal:ncal+800]
        ck,_=split_run(y_k,mu_k,ic,it,alpha)
        nk,_=norm_run(y_k,mu_k,sd_k,ic,it,alpha,1e-3)
        mk,_,_=mondrian_run(y_k,mu_k,ic,it,alpha,5)
        reps.append((ck,nk,mk))
    arr=np.array(reps)
    rows.append(dict(cal_frac=frac, n_cal=ncal,
        split_mean=round(arr[:,0].mean(),4), split_sd=round(arr[:,0].std(),4),
        norm_mean=round(arr[:,1].mean(),4), norm_sd=round(arr[:,1].std(),4),
        mond_mean=round(arr[:,2].mean(),4), mond_sd=round(arr[:,2].std(),4)))
cal_df=pd.DataFrame(rows); cal_df.to_csv("../results/sens_calsize.csv", index=False)
print("=== (b) CALIBRATION-SIZE SENSITIVITY (log_k, mean±sd over 20 splits) ==="); print(cal_df.to_string(index=False)); print()

# ---------- (c) MONDRIAN GROUP COUNT ----------
rows=[]
for nb in [2,3,5,8,10,15]:
    ck,wk,wsk = mondrian_run(y_k,mu_k,i_cal,i_te,alpha,nb)
    cp,wp,wsp = mondrian_run(y_p,mu_p,i_cal,i_te,alpha,nb)
    rows.append(dict(n_groups=nb, cov_logk=round(ck,4), worst_logk=round(wsk,4),
                     cov_poro=round(cp,4), worst_poro=round(wsp,4)))
mon_df=pd.DataFrame(rows); mon_df.to_csv("../results/sens_mondrian.csv", index=False)
print("=== (c) MONDRIAN GROUP-COUNT ABLATION ==="); print(mon_df.to_string(index=False))
