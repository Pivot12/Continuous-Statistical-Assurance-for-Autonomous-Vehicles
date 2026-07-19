# Appendix A.2 - Evidence accumulation: informed priors move the miles wall (Fig A2)
# w=0 reproduces RAND 275M exactly; w=0.2 x 500M sim-miles -> 175M; w=0.5 -> 25M.
import numpy as np
from scipy.stats import gamma as G
Rh=1.09e-8                 # human fatality rate per mile (Kalra & Paddock 2016)
S=500e6                    # validated simulation miles (zero events)
miles=np.linspace(1e6,320e6,600)
for w in [0.0,0.2,0.5]:    # credibility weight on simulation evidence
    ub=np.array([G.ppf(0.95,1,scale=1/(w*S+m)) for m in miles])   # 95% credible UB, zero-event Gamma-Poisson
    c=miles[ub<Rh]
    print('w=%.1f -> crosses human rate at %s'%(w,'%.0fM miles'%(c[0]/1e6) if len(c) else 'never (in range)'))
