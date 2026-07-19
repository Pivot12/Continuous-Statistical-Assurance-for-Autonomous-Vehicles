"""Rebuild FigA1/A2/A5 plot bodies (no header) with annotations moved clear of curves."""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gamma as G, binom
NAVY='#1E3A5F'; GREEN='#63A96F'; AMBER='#E3A64A'
plt.rcParams['axes.grid']=True

# ---------- A1 body ----------
rng=np.random.default_rng(42)
N=200000
d=rng.uniform(0,1,N); p_true=1-0.25*d**1.5; unsafe=p_true<0.90
tau=0.95; kappa=1.0
bmaxs=np.linspace(0,0.15,16); cn=[];cc=[];rn=[];rc=[]
for bm in bmaxs:
    conf=np.clip(p_true+bm*d+rng.normal(0,0.02,N),0,0.9999)
    bins=np.linspace(0,1,11); idx=np.digitize(conf,bins)-1; ece=0.0
    for b in range(10):
        m=idx==b
        if m.sum(): ece+=m.mean()*abs(conf[m].mean()-p_true[m].mean())
    m=(conf>tau-0.05)&(conf<tau+0.05)
    bplus=max(0,(conf[m]-p_true[m]).mean()) if m.sum() else 0
    tau_c=tau+kappa*(ece+bplus)
    A=conf>=tau; C=conf>=tau_c
    cn.append(unsafe[A].mean()*100); cc.append(unsafe[C].mean()*100 if C.sum() else 0)
    rn.append((A&~unsafe).sum()/(~unsafe).sum()*100); rc.append((C&~unsafe).sum()/(~unsafe).sum()*100)

fig,(a1,a2)=plt.subplots(1,2,figsize=(9.2,5.4),dpi=180)
a1.plot(bmaxs,cn,color=AMBER,lw=3.2,label='Naive τ_act')
a1.plot(bmaxs,cc,color=GREEN,lw=3.2,label='Corrected τ′ = τ+κ(ECE+b⁺)')
a1.set_title('Contamination of the ACT zone',fontsize=13,fontweight='bold')
a1.set_xlabel('Boundary overconfidence b_max',fontsize=12)
a1.set_ylabel('Unsafe states inside ACT zone (%)',fontsize=12)
a1.legend(fontsize=10,loc='upper left'); a1.grid(alpha=0.3); a1.tick_params(labelsize=11)
a2.plot(bmaxs,rn,color=AMBER,lw=3.2)
a2.plot(bmaxs,rc,color=GREEN,lw=3.2)
a2.set_title('Cost: autonomy retained (κ trades off)',fontsize=13,fontweight='bold')
a2.set_xlabel('Boundary overconfidence b_max',fontsize=12)
a2.set_ylabel('Safe states retained in ACT (%)',fontsize=12)
a2.grid(alpha=0.3); a2.tick_params(labelsize=11)
# annotation: text fully right of the steep descent (curve hits 0 at ~0.07)
a2.annotate('gate withdraws autonomy\nentirely (conservative\nfailure mode)',
            xy=(0.082,2.0), xytext=(0.112,22), fontsize=10.5, color=GREEN,
            ha='center', va='bottom',
            arrowprops=dict(arrowstyle='->',color=GREEN,lw=1.4))
fig.tight_layout()
fig.savefig('/tmp/nsA/A1_body.png',bbox_inches='tight'); plt.close(fig)

# ---------- A2 body ----------
Rh=1.09e-8; S=500e6
miles=np.linspace(1e6,320e6,600)
fig=plt.figure(figsize=(9,5.6),dpi=180); ax=fig.add_subplot(111)
cfg=[(0.0,AMBER,'w = 0 (frequentist; reproduces RAND 275M)',275,3.05),
     (0.2,GREEN,'w = 0.2 × 500M sim-miles',175,2.45),
     (0.5,NAVY,'w = 0.5 × 500M sim-miles',25,1.85)]
for w,col,lab,cross,laby in cfg:
    ub=np.array([G.ppf(0.95,1,scale=1/(w*S+m)) for m in miles])
    ax.plot(miles/1e6,ub/Rh,color=col,lw=3,label=lab)
    ax.axvline(cross,color=col,ls=':',lw=1.8)
    ax.text(cross-6,laby,f'{cross}M',color=col,fontsize=13,fontweight='bold',ha='right')
ax.axhline(1,color='#555',ls='--',lw=1.8)
# label moved into the clear band above the dashed line (green is >1.2 here, navy <0.95)
ax.text(60,1.06,'parity with human rate',fontsize=11,color='#555',ha='left',va='bottom')
ax.set_xlim(0,320); ax.set_ylim(0,4)
ax.set_xlabel('Real-world fleet miles (millions, zero fatalities)',fontsize=12)
ax.set_ylabel('95% upper bound / human rate',fontsize=12)
ax.legend(fontsize=10.5,loc='upper right'); ax.grid(alpha=0.3); ax.tick_params(labelsize=11)
fig.tight_layout()
fig.savefig('/tmp/nsA/A2_body.png',bbox_inches='tight'); plt.close(fig)

# ---------- A5 body ----------
pf=0.95; pu=0.40
ns=np.arange(1,16)
err_pass=[]; err_fail=[]
for n in ns:
    best=(1,1,1)
    for t in range(1,n+1):
        a=binom.sf(t-1,n,pu); b=binom.cdf(t-1,n,pf)
        if max(a,b)<max(best[0],best[1]) or (max(a,b)==max(best[0],best[1]) and a+b<best[0]+best[1]):
            best=(a,b,t)
    err_pass.append(best[0]*100); err_fail.append(best[1]*100)
n5=[int(n) for n,a,b in zip(ns,err_pass,err_fail) if a<5 and b<5][0]
fig=plt.figure(figsize=(9,5.6),dpi=180); ax=fig.add_subplot(111)
ax.plot(ns,err_pass,'o-',color=AMBER,lw=2.5,label='Unfaithful narration accepted (%)')
ax.plot(ns,err_fail,'o-',color=GREEN,lw=2.5,label='Faithful narration rejected (%)')
ax.axhline(5,color='#555',ls='--',lw=1.5)
# label moved to the flat right-hand tail (both curves < 2% there)
ax.text(14.8,6.0,'5% error budget',fontsize=10.5,color='#555',ha='right',va='bottom')
ax.axvline(n5,color=NAVY,ls=':',lw=1.5)
ax.text(n5+0.15,40,'n = %d replays'%n5,color=NAVY,fontsize=11,fontweight='bold')
ax.set_xlabel('Independent perturbation replays per incident',fontsize=12)
ax.set_ylabel('Error rate (%)',fontsize=12); ax.set_ylim(0,60)
ax.legend(fontsize=10)
ax.grid(alpha=0.3); ax.tick_params(labelsize=11)
fig.tight_layout()
fig.savefig('/tmp/nsA/A5_body.png',bbox_inches='tight'); plt.close(fig)
print('bodies done')
