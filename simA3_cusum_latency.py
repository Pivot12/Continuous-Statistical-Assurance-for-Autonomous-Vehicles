# Appendix A.3 - CUSUM detection latency for a doubled SPI rate (Fig A3)
# ~44 days for 10k mi/day pilot vs ~3 days at 400k mi/day; false alarm ~1/year.
import numpy as np
rng=np.random.default_rng(42)
base=1/100000.0; shift=2.0
k=(shift-1)/np.log(shift)*base       # CUSUM reference value per mile
fleet=np.array([5e3,1e4,2.5e4,5e4,1e5,2e5,4e5,8e5])   # miles/day
def arl0(mpd,h,n=200):
    ds=[]
    for _ in range(n):
        S=0;day=0
        while day<3000:
            day+=1; S=max(0,S+rng.poisson(base*mpd)-k*mpd)
            if S>h: break
        ds.append(day)
    return np.median(ds)
def delay(mpd,h,n=400):
    ds=[]
    for _ in range(n):
        S=0;day=0
        while day<2000:
            day+=1; S=max(0,S+rng.poisson(shift*base*mpd)-k*mpd)
            if S>h: ds.append(day); break
    return np.median(ds)
for mpd in fleet:
    lo,hi=0.1,60
    for _ in range(18):
        mid=(lo+hi)/2
        if arl0(mpd,mid)<365: lo=mid
        else: hi=mid
    print('%6.0fk mi/day -> median detection %5.1f days'%(mpd/1e3,delay(mpd,(lo+hi)/2)))
