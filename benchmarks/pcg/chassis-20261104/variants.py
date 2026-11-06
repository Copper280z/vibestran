# Does constant fail because of the statically determinate support or the soft springs?
import os, re, subprocess, sys, pathlib, json
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/vibestran-pcg-perf/tools"))
import amg_model_suite as s, io
D = pathlib.Path(os.path.expanduser("~/pcg-perf/chassis-20261104")); W = pathlib.Path("/tmp/amg-chassis-var"); W.mkdir(exist_ok=True)
PM = 540
b = io.StringIO(); s.tube_chassis(b, PM); base = b.getvalue()
joints, tubes, corners, supports = s.chassis_layout()
ids = {n: i + 1 for i, n in enumerate(joints)}
LINK = next(l for l in base.splitlines() if l.startswith("PBAR,4,"))
def stiff(t): return re.sub(r"^PBAR,5,.*$", LINK.replace("PBAR,4,", "PBAR,5,"), t, flags=re.M)
def clamp(t):
    # The torsion-rig support: rear suspension pickups clamped, wheel loads at the front only.
    lines = [l for l in t.splitlines() if not l.startswith("SPC1")]
    rear = {("F",7,1),("F",7,-1),("F",8,1),("F",8,-1),("T",7,1),("T",7,-1),("T",8,1),("T",8,-1)}
    forces = [i for i,l in enumerate(lines) if l.startswith("FORCE")]
    for i in forces[2:]: lines[i] = None
    lines = [l for l in lines if l is not None and l != "ENDDATA"]
    lines += [f"SPC1,1,123456,{ids[n]}" for n in sorted(rear, key=str)] + ["ENDDATA"]
    return "\n".join(lines) + "\n"
variants = {"as-built": base, "stiff-links": stiff(base), "rear-clamped": clamp(base), "rear-clamped+stiff": clamp(stiff(base))}
env = dict(os.environ, OMP_NUM_THREADS="8")
results = []
for name, text in variants.items():
    deck = W / f"{name}.bdf"; deck.write_text(text)
    for modes in ("rigid", "constant"):
        log, f06 = W / f"{name}-{modes}.log", W / f"{name}-{modes}.f06"
        cmd = ["taskset","-c","0-7",str(D/"vibestran"),"--backend=cuda-pcg",f"--log-file={log}",f"--amg-modes={modes}",str(deck),str(f06)]
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, timeout=200); status = r.returncode
        except subprocess.TimeoutExpired: status = "timeout"
        text_log = log.read_text() if log.exists() else ""
        m = re.findall(r"iterations=(\d+), estimated error=(\S+)", text_log)
        row = {"variant": name, "modes": modes, "status": status, "iter_err": m}
        results.append(row); print(row, flush=True)
(D/"variants.json").write_text(json.dumps(results, indent=1))
import shutil; shutil.rmtree(W)
