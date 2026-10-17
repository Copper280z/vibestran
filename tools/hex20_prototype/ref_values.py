import numpy as np
import pathlib
REPO = str(pathlib.Path(__file__).resolve().parents[2])
from run_deck import solve
D=REPO+"/tests/integration/data/hexa20_reference/"
tw=(25,38,63,76,101,114,127,140,165,178,203,216,241)
cases=[("CHEXA20_pinched_hemisphere_n08","mean",0,(1,226,307)),("CHEXA20_pinched_hemisphere_n12","mean",0,(1,482,651)),
("CHEXA20_pinched_cylinder_n08","mean",2,(1,226,307)),("CHEXA20_pinched_cylinder_n12","mean",2,(1,482,651)),
("CHEXA20_scordelis_lo","sum",2,(209,298,515)),("CHEXA20_twisted_beam_inplane","sum",2,tw),
("CHEXA20_twisted_beam_outofplane","sum",1,tw),("CHEXA20_plate_ss","sum",2,(1,226,307))]
for f,k,c,g in cases:
    m,idx,U,_=solve(D+f+".bdf","asnl2")
    v=[U[idx[n]][c] for n in g]; v=np.mean(v) if k=="mean" else np.sum(v)
    print(f'{f} {k} {c} {v:.12e}')
