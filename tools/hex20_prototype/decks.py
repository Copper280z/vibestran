import sys, numpy as np
import pathlib
REPO = str(pathlib.Path(__file__).resolve().parents[2])
from run_deck import solve
B=REPO+"/MYSTRAN/hexa_msc_comparison/"
V=sys.argv[1].split(",")
def sumc(grids,c): return lambda m,idx,U: sum(U[idx[g]][c] for g in grids)
def meanc(grids,c): return lambda m,idx,U: np.mean([U[idx[g]][c] for g in grids])
tw=(25,38,63,76,101,114,127,140,165,178,203,216,241)
cases=[("shell/CHEXA20_scordelis_lo.bdf",sumc((209,298,515),2),-0.9072),
 ("shell/CHEXA20_twisted_beam_inplane.bdf",sumc(tw,2),0.070512),
 ("shell/CHEXA20_twisted_beam_outofplane.bdf",sumc(tw,1),0.022802),
 ("shell/CHEXA20_plate_ss.bdf",sumc((1,226,307),2),-0.133083),
 ("shell/CHEXA20_plate_clamped.bdf",sumc((1,226,307),2),-0.0414519)]
for n,(g) in [("04",(1,66,91)),("08",(1,226,307)),("12",(1,482,651)),("16",(1,834,1123))]:
    cases.append((f"convergence/CHEXA20_pinched_hemisphere_n{n}.bdf",meanc(g,0),0.094))
    cases.append((f"convergence/CHEXA20_pinched_cylinder_n{n}.bdf",meanc(g,2),-1.8248e-5))
def patch(m,idx,U):
    err=0
    for n,i in idx.items():
        x,y,z=m.nodes[n].get_position()
        ex=1e-3*np.array([2*x+y+z,x+2*y+z,x+y+2*z])/2
        err=max(err,abs(U[i]-ex).max())
    return err/1e-3
def radial(m,idx,U):
    vals=[]
    for n,i in idx.items():
        x,y,z=m.nodes[n].get_position(); r=np.hypot(x,y)
        if abs(r-1)<1e-6: vals.append((U[i][0]*x+U[i][1]*y)/r)
    return np.mean(vals)
cases+= [("robustness/patch_20_default.bdf",patch,None),
 ("robustness/cyl_20_0.3_default.bdf",radial,1.906667e-6),
 ("robustness/cyl_20_0.4999_default.bdf",radial,1.999967e-6)]
print(f"{'deck':48s}"+"".join(f"{v:>10s}" for v in V))
for f,q,ref in cases:
    row=[]
    for v in V:
        m,idx,U,_=solve(B+f,v); val=q(m,idx,U)
        row.append(f"{100*val/ref:9.1f}%" if ref else f"{val:10.1e}")
    print(f"{f:48s}"+"".join(row), flush=True)
