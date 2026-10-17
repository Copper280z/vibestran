import itertools, numpy as np
from hex20 import NAT, Element, shape, gauss
rng=np.random.default_rng(7)
V=["as","asn","asnl","asnl2"]
def curved_distorted():
    X=NAT*[2,1.5,1]+0.12*rng.uniform(-1,1,NAT.shape)
    for i in range(8,20): X[i]+=0.3*rng.uniform(-1,1,3)   # curved, off-centre midside nodes
    return X
def shell_segment(R=10,t=0.04,ang=0.2):   # hemisphere-like: curved both ways, aspect ~ R*ang/t
    X=[]
    for xi,eta,ze in NAT:
        th=ang/2*xi; ph=ang/2*eta; r=R+t/2*ze
        X.append([r*np.cos(ph)*np.cos(th), r*np.cos(ph)*np.sin(th), r*np.sin(ph)])
    return np.array(X)
shapes={"curved-distorted":curved_distorted(),"shell-segment":shell_segment()}
# 1) constant strain (patch test at element level) -- strain at every Gauss point
def strains(el,u):
    # rebuild Bbar per GP via K-free path: reuse internals by finite trick: energy check instead
    return None
print("1) patch: u = A x must give K u = consistent nodal forces of constant stress, checked by")
print("   energy: u_lin' K u_lin == V * eps' D eps  and  K (u_lin) orthogonal to all rigid modes")
for name,X in shapes.items():
    vol=sum(w*np.linalg.det(X.T@shape(x)[1]) for x,w in zip(*gauss(4)))
    for v in V:
        el=Element(X,1e6,0.3,v); K=el.stiffness(); err=0
        for k in range(6):
            e=np.zeros(6); e[k]=1e-3
            E=np.array([[e[0],e[3]/2,e[5]/2],[e[3]/2,e[1],e[4]/2],[e[5]/2,e[4]/2,e[2]]])
            u=(X@E.T).ravel()
            ex=vol*e@el.D@e
            err=max(err,abs(u@K@u-ex)/ex)
            # mixed energy with another field: u2'K u = V eps2' D eps
            for k2 in range(6):
                e2=np.zeros(6); e2[k2]=1e-3
                E2=np.array([[e2[0],e2[3]/2,e2[5]/2],[e2[3]/2,e2[1],e2[4]/2],[e2[5]/2,e2[4]/2,e2[2]]])
                u2=(X@E2.T).ravel(); ex2=vol*e2@el.D@e
                err=max(err,abs(u2@K@u-ex2)/abs(vol*e@el.D@e))
        Kf=Element(X,1e6,0.3,"full").stiffness(); e2=0
        for k in range(6):
            e=np.zeros(6); e[k]=1e-3
            E=np.array([[e[0],e[3]/2,e[5]/2],[e[3]/2,e[1],e[4]/2],[e[5]/2,e[4]/2,e[2]]])
            u=(X@E.T).ravel(); e2=max(e2,abs(K@u-Kf@u).max()/abs(Kf@u).max())
        print(f"   {name:17s} {v:5s} energy {err:.1e}  nodal forces of constant stress {e2:.1e}")
# 2) rank with scale-aware tolerance: 6 zero modes, lambda_7 relative to the full element
print("2) spectrum (lambda_7 / lambda_7(full), zero count)")
for name,X in list(shapes.items())+[("needle",NAT*[10,0.5,0.5]),("slab",NAT*[5,5,0.25])]:
    for nu in (0.3,0.4999):
        evf=np.linalg.eigvalsh(Element(X,1e6,nu,"full").stiffness())
        out=[]
        for v in V:
            ev=np.linalg.eigvalsh(Element(X,1e6,nu,v).stiffness())
            nz=(abs(ev)<1e-10*evf[6]).sum()
            out.append(f"{v}: {nz}z {ev[6]/evf[6]:.2e}")
        print(f"   {name:17s} nu={nu}: "+"  ".join(out))
# 3) invariance
print("3) invariance (max |dK|/max|K|)")
X=curved_distorted()
def perms():
    for P in itertools.permutations(range(3)):
        for s in itertools.product((1,-1),repeat=3):
            R=np.zeros((3,3))
            for i in range(3): R[i,P[i]]=s[i]
            if np.linalg.det(R)>0: yield R
for v in V:
    K0=Element(X,1e6,0.3,v).stiffness(); worst=0
    for R in perms():
        perm=[int(np.flatnonzero((abs(NAT-(R@NAT[i]))<1e-12).all(1))[0]) for i in range(20)]
        K=Element(X[perm],1e6,0.3,v).stiffness()
        dof=np.array([3*p+k for p in perm for k in range(3)])
        Kb=np.zeros_like(K0); Kb[np.ix_(dof,dof)]=K
        worst=max(worst,abs(Kb-K0).max()/abs(K0).max())
    Q,_=np.linalg.qr(rng.normal(size=(3,3))); Q*=np.sign(np.linalg.det(Q))
    Kr=Element(X@Q.T,1e6,0.3,v).stiffness(); T=np.kron(np.eye(20),Q)
    print(f"   {v:5s} 24 relabelings {worst:.1e}  rotation {abs(T.T@Kr@T-K0).max()/abs(K0).max():.1e}")
