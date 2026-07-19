import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import poisson, binom
rng=np.random.default_rng(42)
NAVY='#1E3A5F'; GREEN='#63A96F'; AMBER='#E3A64A'; LBLUE='#A8C8E8'
def banner(fig,title,sub):
    ax=fig.add_axes([0,0.90,1,0.10]); ax.axis('off')
    ax.add_patch(Rectangle((0,0),1,1,color=NAVY,transform=ax.transAxes))
    ax.text(0.02,0.68,title,color='white',fontsize=14,fontweight='bold',va='center')
    ax.text(0.02,0.24,sub,color=LBLUE,fontsize=10.5,va='center')

# ---- A4: value of the aggregate return channel ----
base=1/100000.0; elev=1.5; alpha=0.05; NREP=4000
miles=np.array([0.25,0.5,1,2,4,8,16])*1e6
pow_bench=[]; pow_self=[]
for M in miles:
    lam0=base*M
    crit=poisson.ppf(1-alpha,lam0)     # one-sample test vs known benchmark
    pow_bench.append((rng.poisson(elev*lam0,NREP)>crit).mean())
    # self-baselined: compare this period vs own equal-length prior period (binomial conditional test)
    det=0
    for _ in range(NREP):
        x_prev=rng.poisson(lam0); x_now=rng.poisson(elev*lam0)
        n=x_prev+x_now
        if n>0 and binom.sf(x_now-1,n,0.5)<alpha: det+=1
    pow_self.append(det/NREP)
fig=plt.figure(figsize=(9,5.6),dpi=180)
banner(fig,'Figure A4. What the aggregate return channel is worth to a small fleet',
       'Power to confirm a 1.5x elevation of a 1/100k-mile SPI at alpha=0.05; 4,000 Monte Carlo runs/point')
ax=fig.add_axes([0.10,0.13,0.86,0.64])
ax.plot(miles/1e6,np.array(pow_bench)*100,'o-',color=GREEN,lw=2.5,label='Anchored to pooled field benchmark (return channel)')
ax.plot(miles/1e6,np.array(pow_self)*100,'o-',color=AMBER,lw=2.5,label='Self-baselined (own prior-period data only)')
ax.axhline(80,color='#555',ls='--',lw=1.5); ax.text(0.27,82,'80% power',fontsize=10,color='#555')
ax.set_xscale('log'); ax.set_xlabel('Fleet miles in the monitoring window (millions)',fontsize=12)
ax.set_ylabel('Power to detect the elevation (%)',fontsize=12)
ax.legend(fontsize=10,loc='lower right'); ax.grid(alpha=0.3,which='both'); ax.tick_params(labelsize=11)
fig.savefig('/tmp/figs/FigA4.png',bbox_inches='tight'); plt.close(fig)
import numpy as _np
def interp80(p): 
    p=_np.array(p)*100
    for i in range(len(p)-1):
        if p[i]<80<=p[i+1]:
            return miles[i]/1e6*(miles[i+1]/miles[i])**((80-p[i])/(p[i+1]-p[i]))
    return None
print('A4 power:',dict(zip(miles/1e6,np.round(np.array(pow_bench)*100,1))),dict(zip(miles/1e6,np.round(np.array(pow_self)*100,1))))
print('80pct-power miles: bench %.2fM self %.2fM'%(interp80(pow_bench),interp80(pow_self)))

# ---- A5: replay power for Layer 3b ----
pf=0.95; pu=0.40   # per-replay consistency: faithful vs unfaithful narration
ns=np.arange(1,16)
err_pass=[]; err_fail=[]; ts=[]
for n in ns:
    best=(1,1,1)
    for t in range(1,n+1):
        a=binom.sf(t-1,n,pu)        # unfaithful passes (consistent >= t)
        b=binom.cdf(t-1,n,pf)       # faithful rejected
        if max(a,b)<max(best[0],best[1]) or (max(a,b)==max(best[0],best[1]) and a+b<best[0]+best[1]):
            best=(a,b,t)
    err_pass.append(best[0]*100); err_fail.append(best[1]*100); ts.append(best[2])
fig=plt.figure(figsize=(9,5.6),dpi=180)
banner(fig,'Figure A5. Counterfactual replays needed to expose unfaithful narration',
       'Per-replay consistency: faithful 0.95 vs unfaithful 0.40 (measured VLA range); optimal accept-threshold per n')
ax=fig.add_axes([0.10,0.13,0.86,0.64])
ax.plot(ns,err_pass,'o-',color=AMBER,lw=2.5,label='Unfaithful narration accepted (%)')
ax.plot(ns,err_fail,'o-',color=GREEN,lw=2.5,label='Faithful narration rejected (%)')
ax.axhline(5,color='#555',ls='--',lw=1.5); ax.text(1.1,5.8,'5% error budget',fontsize=10,color='#555')
n5=[n for n,a,b in zip(ns,err_pass,err_fail) if a<5 and b<5][0]
ax.axvline(n5,color=NAVY,ls=':',lw=1.5); ax.text(n5+0.15,40,'n = %d replays'%n5,color=NAVY,fontsize=11,fontweight='bold')
ax.set_xlabel('Independent perturbation replays per incident',fontsize=12)
ax.set_ylabel('Error rate (%)',fontsize=12); ax.set_ylim(0,60)
ax.legend(fontsize=10); ax.grid(alpha=0.3); ax.tick_params(labelsize=11)
fig.savefig('/tmp/figs/FigA5.png',bbox_inches='tight'); plt.close(fig)
print('A5: first n with both errors <5%%: %d (threshold %d); errors %.1f%%/%.1f%%'%(n5,ts[n5-1],err_pass[n5-1],err_fail[n5-1]))
