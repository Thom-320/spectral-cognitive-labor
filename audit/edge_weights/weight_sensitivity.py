"""Sensibilidad del optimo de conductancia al peso de las aristas diagonales.

Responde a la pregunta de si un peso no uniforme sobre el grafo espacial cambia
la particion optima. Solo geometria: no usa datos de participantes.
Uso: python audit/edge_weights/weight_sensitivity.py
"""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
n=8; N=64
E=[]  # (u,v,diagonal?)
for r in range(n):
    for c in range(n):
        for dr,dc in ((0,1),(1,0),(1,1),(1,-1)):
            rr,cc=r+dr,c+dc
            if 0<=rr<n and 0<=cc<n: E.append((r*n+c, rr*n+cc, dr!=0 and dc!=0))
def wts(w): return np.array([w if d else 1.0 for _,_,d in E])
def deg(w):
    d=np.zeros(N); 
    for (u,v,dg),x in zip(E,wts(w)): d[u]+=x; d[v]+=x
    return d
y,x=np.indices((n,n)); x=x.ravel(); y=y.ravel()
cands={'LR (mitades)':x<4,'TB (mitades)':y<4,'Diagonal':(x+y)<7,'IN 6x6 / OUT':(x>0)&(x<7)&(y>0)&(y<7),
       'IN 4x4 / OUT':(x>1)&(x<6)&(y>1)&(y<6),'Escalera (2 col + 2 col desplazadas)':((x<4)&(y<4))|((x<4)&(y>=4)),
       'Cuadrante 4x4':(x<4)&(y<4)}
def h(S,w):
    d=deg(w); cut=sum(x_ for (u,v,_),x_ in zip(E,wts(w)) if S[u]!=S[v]); vol=d[S].sum(); volb=d.sum()-vol
    return cut/min(vol,volb) if min(vol,volb)>0 else np.nan
print("w = peso de la arista diagonal\n")
print(f"{'particion':38s}" + "".join(f"{w:>10}" for w in [1.0,0.9,0.8,0.7071,0.6,0.5,0.3]))
for name,S in cands.items():
    print(f"{name:38s}" + "".join(f"{h(S,w):10.4f}" for w in [1.0,0.9,0.8,0.7071,0.6,0.5,0.3]))
def global_opt(w,tl=300):
    ww=wts(w); d=deg(w); M=len(E); lam_hist=[]
    lam=min(h(S,w) for S in cands.values())
    for it in range(12):
        rows=[];lb=[];ub=[]
        for k,(u,v,_) in enumerate(E):
            for a,b in ((u,v),(v,u)):
                row=np.zeros(N+M); row[a]=1; row[b]=-1; row[N+k]=-1; rows.append(row); lb.append(-np.inf); ub.append(0)
        row=np.zeros(N+M); row[:N]=d; rows.append(row); lb.append(1e-6); ub.append(d.sum()/2)
        cost=np.concatenate([-lam*d, ww])
        res=milp(cost,constraints=LinearConstraint(np.array(rows),lb,ub),integrality=np.ones(N+M),bounds=Bounds(0,1),options={"time_limit":tl,"mip_rel_gap":0})
        xs=res.x[:N]>0.5
        cut=sum(x_ for (u,v,_),x_ in zip(E,ww) if xs[u]!=xs[v]); vol=d[xs].sum()
        newlam=cut/min(vol,d.sum()-vol)
        if res.fun>=-1e-9 or abs(newlam-lam)<1e-12:
            return lam, xs, res.status
        lam=newlam
    return lam, xs, res.status
print("\nOptimo global (MILP, Dinkelbach):")
for w in [1.0,0.7071,0.5]:
    hstar,S,st=global_opt(w)
    grid=S.reshape(8,8).astype(int)
    match=[k for k,v in cands.items() if np.array_equal(v,S) or np.array_equal(~v,S)]
    print(f"  w={w:<7} h*={hstar:.5f} status={st} |S|={S.sum()} coincide con: {match if match else 'otra'}")
    print("   " + " ".join("".join(map(str,row)) for row in grid))
