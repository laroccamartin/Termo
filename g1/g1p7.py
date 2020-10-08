#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRANDES NUMEROS
"""

#IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from numpy import array as ar,pi
from numpy.random import random as rnd
from mpl_toolkits.mplot3d import Axes3D
from numpy import sin,cos,dot,arccos

#%% 
"""
LA DENSIDAD ES f(tita,phi) = sen(tita)/4/pi
LA DENSIDAD MARGINAL DE TITA ES f_tita (tita)= sen(tita)/2 
LA DISTRIBUCION MARGINAL ACUMULADA DE TITA ES F_tita (tita)= (1-cos(tita))/2
POR LO TANTO, SAMPLEO UNIFORME R EN [0,PI] Y LUEGO MAPA TITA=ARCCOS(1-2R)
"""

NUM=100000
usampled =rnd(NUM)
titas=arccos(1-2*usampled) 
phis=2*pi*rnd(NUM)  # CON DISTRIBUCION MARGINAL UNIFORME

#%% CALCULO VERSORES RANDOM

def mapa(tita,phi):
    x=sin(tita)*cos(phi)
    y=sin(tita)*sin(phi)
    z=cos(tita)
    return ar([x,y,z])

vs=[]
v0=mapa(titas[0],phis[0])
ps=[]
for k in range(1,NUM):
    tita=titas[k]
    phi=phis[k]
    v = mapa(tita,phi)
    vs.append(v)
    ps+=[v0@v]

#%% 1 GRAFICO LOS VECTORES

TRUNC=1000 #TRUNCO PARA NO FUNDIR LA MAQUINA
soa=ar([[0,0,0]+list(v) for v in vs[:TRUNC]])

X, Y, Z, U, V, W = zip(*soa)
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(X, Y, Z, U, V, W,color="black")
ax.set_xlim([-1, 1])
ax.set_ylim([-1,1])
ax.set_zlim([-1, 1])
x=[-1,0,1]
ax.set_xticks(x)
ax.set_yticks(x)
ax.set_zticks(x)

plt.savefig("g1p7_quivers.png")
plt.savefig("g1p7_quivers.svg")


#%% 2 GRAFICO LAS PROYECCIONES
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15) 

plt.figure(1,figsize=(8,6))

d=1e-2
x=np.arange(-1-1e-4,1+d,d)
plt.hist(ps,bins=x,density=True)
d2=.5
x2=np.arange(-1,1+d2,d2)
plt.xticks(x2)

plt.xlabel("$\epsilon$",fontsize=20)
plt.ylabel("distribucion",fontsize=20)
plt.savefig("g1p7_distrib.png")
plt.savefig("g1p7_distrib.svg")

#%% 3 GRAFICO LAS DISTRIB PHI Y TITA
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15) 

plt.figure(1,figsize=(8,6))

lb=0;ub=2*pi
d=(ub-lb)*1e-3
x=np.arange(lb-1e-4,ub+d,d)
plt.hist(phis,bins=x,density=True,color="g")

d=(ub-lb)*.5
x2=np.arange(lb,ub+d2,d2)
plt.xticks(x2)

plt.ylabel("distribucion",fontsize=20)
plt.xlabel("$\phi$",fontsize=20)
plt.savefig("g1p7_phi_v2.png")
plt.savefig("g1p7_phi_v2.svg")
#%% 3 GRAFICO LAS DISTRIB PHI Y TITA
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15) 

plt.figure(1,figsize=(8,6))

lb=0;ub=pi
d=(ub-lb)*1e-3
x=np.arange(lb-1e-4,ub+d,d)
plt.hist(titas,bins=x,density=True,color="r")

d=(ub-lb)*.5
x2=np.arange(lb,ub+d2,d2)
plt.xticks(x2)

plt.ylabel("distribucion",fontsize=20)
plt.xlabel("tita",fontsize=20)
plt.savefig("g1p7_tita_v2.png")
plt.savefig("g1p7_tita_v2.svg")

#%% BONUS: PROYECCIONES EN 2D

NUM=100000

vs=[]
for k in range(NUM):
    phi=rnd(1)[0]*2*pi
    
    x=cos(phi)
    y=sin(phi)
    
    v = [x,y]
    
    vs.append(v)
    
ps=[np.dot(vs[0],v) for v in vs] #PROYECCIONES

#%% 
plt.figure(1,figsize=(8,6))

d=1e-2
x=np.arange(-1-1e-4,1+d,d)
plt.hist(ps,bins=x,density=True)
d2=.5
x2=np.arange(-1,1+d2,d2)
plt.xticks(x2)

f=1/pi/sin(arccos(x))
plt.plot(x[:-3],f[:-3],"--",color="black",linewidth=3,label="$f(\epsilon)=1/\pi/sin(arccos(\epsilon))$")

plt.legend()
plt.xlabel("$\epsilon$",fontsize=20)
plt.ylabel("distribucion",fontsize=20)
plt.savefig("g1p7_distrib2.png")

#%%

xvs=vs[:TRUNC]
# defining necessary arrays 
x_coordinate = np.zeros(len(xvs))
y_coordinate = np.zeros(len(xvs))
x_direction = ar(xvs)[:,0]
y_direction = ar(xvs)[:,1]
  

fig = plt.figure()
ax = fig.add_subplot(111)
ax.quiver(x_coordinate, y_coordinate, x_direction, y_direction, units = 'xy', scale = 1)
plt.axis('equal')
plt.xticks(range(-2,3))
plt.yticks(range(-2,3))
plt.grid()

plt.savefig("g1p7_quivers2.png")


