import json, os, re, subprocess, sys, time
sys.path.insert(0, os.path.expanduser("~/vibestran-pcg-perf/tools"))
import amg_model_suite as suite
OUT = os.path.expanduser("~/pcg-perf/cpucycle-20261106"); TMP = "/tmp/cpucycle"
LIB = "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/libcudss/12"
BIN = os.environ.get("CYCLE_BIN", os.path.expanduser("~/vibestran-pcg-perf/build/vibestran"))
RES = os.environ.get("CYCLE_RES", f"{OUT}/runs.jsonl")
def deck_for(name, size):
    path = f"{TMP}/{name}-{size}.bdf"
    if not os.path.exists(path): suite.write_deck(name, size, path)
    return path
def run(tag, deck, config, threads=8, timeout=200, backend="cpu-pcg", cores=None, extra=None, label=None):
    env = {**os.environ, "LD_LIBRARY_PATH": LIB, "OMP_NUM_THREADS": str(threads), **{k: str(v) for k, v in config.items()}}
    label = label or ",".join(k.replace("VIBESTRAN_PROTO_","").lower()+"="+str(v) for k, v in config.items()) or backend
    name = f"{tag}-{re.sub("[^a-z0-9]+","_",label.lower())}-{threads}"; log, f06 = f"{TMP}/{name}.log", f"{TMP}/{name}.f06"
    if os.path.exists(log): os.remove(log)
    mask = cores or f"0-{threads-1}"
    magnetic = tag.startswith("magnetic")
    cmd = ["/usr/bin/time", "-f", "TIME %e %U %S %M", "taskset", "-c", mask, BIN, f"--backend={backend}", f"--log-file={log}"] + ([] if magnetic else ["--no-quality-checks"]) + (extra or []) + [deck, f06]
    start = time.monotonic()
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout); rc = p.returncode; err = p.stderr
    except subprocess.TimeoutExpired:
        rc, err = "timeout", ""
    wall = time.monotonic() - start
    t = re.search(r"TIME ([\d.]+) ([\d.]+) ([\d.]+) (\d+)", err or "")
    text = open(log, errors="replace").read() if os.path.exists(log) else ""
    row = {"tag": tag, "label": label, "threads": threads, "rc": rc, "wall": round(wall, 1), "cpu_pct": round(100 * (float(t[2]) + float(t[3])) / max(float(t[1]), .01)) if t else None, "rss_gib": round(int(t[4]) / 2**20, 1) if t else None}
    m = re.search(r"setup=([\d.]+) ms, iteration/refinement=([\d.]+) ms, iterations=(\d+), estimated error=([\d.e+-]+)", text)
    if m: row.update(setup_s=round(float(m[1]) / 1000, 2), iter_s=round(float(m[2]) / 1000, 2), its=int(m[3]), err=float(m[4]))
    m = re.search(r"preconditioner: ([\d.]+) ms, iterative solve: ([\d.]+) ms, iterations: (\d+), corrections: (\d+)", text)
    if m: row.update(setup_s=round(float(m[1]) / 1000, 2), iter_s=round(float(m[2]) / 1000, 2), its=int(m[3]))
    m = re.findall(r"n=(\d+), nnz=(\d+)", text)
    if m: row.update(n=int(m[-1][0]), nnz=int(m[-1][1]))
    m = re.search(r"rows per list ([\d.]+)", text)
    if m: row["rows_per_list"] = float(m[1])
    if "its" in row and row["its"]: row["ms_it"] = round(row["iter_s"] * 1000 / row["its"], 2)
    with open(RES, "a") as out: out.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    keep = os.environ.get("CYCLE_KEEP_LOGS")
    for path in (f06, f06.replace(".f06", ".op2")) + (() if keep else (log,)):
        if os.path.exists(path): os.remove(path)
    return row
