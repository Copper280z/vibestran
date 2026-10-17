import numpy as np
import pathlib
REPO = str(pathlib.Path(__file__).resolve().parents[2])
from pyNastran.bdf.bdf import read_bdf
from hex20 import NAT, Element
rng=np.random.default_rng(7)
def rb(X):  # rigid-body basis
    R=[]
    for k in range(3):
        u=np.zeros((20,3)); u[:,k]=1; R.append(u.ravel())
    for k in range(3):
        w=np.zeros(3); w[k]=1; R.append(np.cross(w,X-X.mean(0)).ravel())
    Q,_=np.linalg.qr(np.array(R).T); return Q
def report(name,X,V=("as","asn","asnl2")):
    Q=rb(X); Pn=np.eye(60)-Q@Q.T  # remove rigid modes exactly -> rank check on complement
    for nu in (0.3,0.4999):
        Kf=Element(X,1e6,nu,"full").stiffness()
        evf=np.linalg.eigvalsh(Pn@Kf@Pn)[6:]
        line=[]
        for v in V:
            K=Element(X,1e6,nu,v).stiffness()
            rig=abs(K@Q).max()/abs(K).max()
            ev=np.linalg.eigvalsh(Pn@K@Pn)[6:]
            line.append(f"{v}: rigid {rig:.0e} min(lam/lam_full) {np.min(ev/evf):.2e}")
        print(f"{name:22s} nu={nu}: "+" | ".join(line))
X=NAT*[2,1.5,1]+0.12*rng.uniform(-1,1,NAT.shape)
for i in range(8,20): X[i]+=0.3*rng.uniform(-1,1,3)
report("curved-distorted",X)
report("needle 20:1:1",NAT*[10,0.5,0.5]); report("slab 10:10:1",NAT*[5,5,0.5])
m=read_bdf(REPO+"/MYSTRAN/hexa_msc_comparison/convergence/CHEXA20_pinched_hemisphere_n08.bdf",xref=True,debug=None)
for eid in list(m.elements)[:: 21]:
    e=m.elements[eid]; X=np.array([m.nodes[n].get_position() for n in e.node_ids])
    report(f"hemisphere n08 e{eid}",X)
