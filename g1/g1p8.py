#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRANDES NUMEROS
"""

# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from numpy import array as ar


# TEST USE OF HIST
s=[2,4,4,5,6,6]
s2=ar(s)/10

# =============================================================================
# plt.hist(s,bins=np.arange(0,11,1))
# =============================================================================
plt.hist(s2,bins=np.arange(-1e-5,1.1,.1),density=True)

#%% EXPERIMENTO

from numpy.random import randint

NREP=10000
# =============================================================================
# NREP=100
# =============================================================================
Ns = [int(N) for N in np.logspace(1,4,4)]; #LIST CON DISTINTOS NUMEROS DE MONEDAS
fss=[]
for N in Ns:
    print("N",N)
    fs=[]
    for k in range(NREP): # REPITO EL EXPERIMENTO NREP VECES
        
        #HAGO EL EXPERIMENTO: TIRAR LA MONEDA N VECES
        f=sum([randint(0,high=2) for k in range(N)])/N #RESULTADO = FRECUENCIA RELATIVA
        fs.append(f)
    fss.append(fs)

#%% PLOTEO

from numpy import pi

Ns=[int(N) for N in Ns]
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

col=["b","g","r","y"]
p=.5
fig,ax = plt.subplots(4,1,sharex=False,figsize=(8,12))
for i,fs in enumerate(fss):
    N=Ns[i]
    mu=p
    sigma=np.sqrt(p*(1-p)/N)
    
    x=np.linspace(p-2*sigma,p+2*sigma,100)

    gauss=1/np.sqrt(2*pi)/sigma*np.exp(-(x-mu)**2/2/sigma**2)
    ax[i].plot(x,gauss,"--",color="black",linewidth=3)
    
    
    step=1/N
    x2=np.arange(-1e-4/N,1+step,step)
    ax[i].hist(fs,bins=x2,density=True,label="N={}".format(N),color=col[i])
    
    num=10
    gap=2*sigma
    ax[i].legend()
    ax[i].set_xlim([p-gap,p+gap])
    
    lb=p-gap
    ub=p+gap
    d=(ub-lb)/5
    tics=np.arange(lb,ub+d,d)
    ax[i].set_xticks([lb,ub])

ax[3].set_xlabel("s",fontsize=20)

ax[0].set_ylabel("$f_S(s)$",fontsize=20)
ax[1].set_ylabel("$f_S(s)$",fontsize=20)
ax[2].set_ylabel("$f_S(s)$",fontsize=20)
ax[3].set_ylabel("$f_S(s)$",fontsize=20)

plt.savefig("g1p8_histo.png")


