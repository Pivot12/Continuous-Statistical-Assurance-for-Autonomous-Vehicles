# Appendix A.1 - Calibration-aware gating under attractor overconfidence (Fig A1)
# Reproduces: naive tau admits 1.3-9.7% unsafe states into ACT; corrected tau' ~0%, conservative failure mode.
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
rng=np.random.default_rng(42)
N=200000
d=rng.uniform(0,1,N)                 # state difficulty
p_true=1-0.25*d**1.5                 # true success probability
unsafe=p_true<0.90
tau=0.95; kappa=1.0
bmaxs=np.linspace(0,0.15,16); cn=[];cc=[];rn=[];rc=[]
for bm in bmaxs:
    conf=np.clip(p_true+bm*d+rng.normal(0,0.02,N),0,0.9999)   # overconfidence grows with difficulty
    bins=np.linspace(0,1,11); idx=np.digitize(conf,bins)-1; ece=0.0
    for b in range(10):
        m=idx==b
        if m.sum(): ece+=m.mean()*abs(conf[m].mean()-p_true[m].mean())
    m=(conf>tau-0.05)&(conf<tau+0.05)
    bplus=max(0,(conf[m]-p_true[m]).mean()) if m.sum() else 0
    tau_c=tau+kappa*(ece+bplus)      # no cap: gate may withdraw autonomy entirely (conservative)
    A=conf>=tau; C=conf>=tau_c
    cn.append(unsafe[A].mean()*100); cc.append(unsafe[C].mean()*100 if C.sum() else 0)
    rn.append((A&~unsafe).sum()/(~unsafe).sum()*100); rc.append((C&~unsafe).sum()/(~unsafe).sum()*100)
print('b_max=0.05: contamination %.1f%%->%.1f%%, retention %.0f%%->%.0f%%'%(cn[5],cc[5],rn[5],rc[5]))
print('b_max=0.10: contamination %.1f%%->%.1f%%, retention %.0f%%->%.0f%%'%(cn[10],cc[10],rn[10],rc[10]))
