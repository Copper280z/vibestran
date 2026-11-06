# Stored against computed transfers, with the built binary, driven by budget.
import json, os, re, sys, shutil
sys.path.insert(0, os.path.expanduser("~/vibestran-pcg-perf/tools"))
import amg_model_suite as suite
OUT = os.path.expanduser("~/pcg-perf/implicit-20261105"); TMP = "/tmp/implicit"; BIN = f"{OUT}/vibestran"
LIB = "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/libcudss/12"
os.makedirs(TMP, exist_ok=True)
def run(tag, deck, modes, budget, label, backend="cuda-pcg", timeout=900, prefix=(), results="runs.jsonl", binary=None):
    env = {**os.environ, "LD_LIBRARY_PATH": LIB, "OMP_NUM_THREADS": "8"}
    env.pop("VIBESTRAN_PCG_DEVICE_MEMORY_MIB", None)
    if budget: env["VIBESTRAN_PCG_DEVICE_MEMORY_MIB"] = str(budget)
    name = f"{tag}-{modes}-{label}-{backend}"; log, f06 = f"{OUT}/{name}.log", f"{TMP}/{name}.f06"
    if os.path.exists(log): os.remove(log)
    cmd = [*prefix, "taskset", "-c", "0-7", binary or BIN, f"--backend={backend}", *([f"--amg-modes={modes}"] if modes else []), "--no-quality-checks", f"--log-file={log}", deck, f06]
    outcome, detail, code, wall, rss = suite.run_and_measure(cmd, env, timeout)
    text = open(log, errors="replace").read() if os.path.exists(log) else ""
    s = re.search(r"setup=([\d.]+) ms, iteration/refinement=([\d.]+) ms, iterations=(\d+), estimated error=([\d.e+-]+)", text)
    split = re.search(r"(\d+) GPU rows, (\d+) CPU rows.*fixed=(\d+) planned=(\d+) budget=(\d+) bytes; (\w+) finest", text)
    measured = re.search(r"planned=\d+ measured=(\d+)", text)
    row = {"tag": tag, "modes": modes, "label": label, "backend": backend, "budget_mib": budget, "outcome": outcome, "wall_s": round(wall, 1), "setup_s": round(float(s[1]) / 1000, 2) if s else None,
        "iter_s": round(float(s[2]) / 1000, 2) if s else None, "its": int(s[3]) if s else None, "error": s[4] if s else None,
        "ms_it": round(float(s[2]) / int(s[3]), 2) if s and int(s[3]) else None, "gpu_rows": int(split[1]) if split else None, "cpu_rows": int(split[2]) if split else None,
        "fixed_mib": round(int(split[3]) / 2**20) if split else None, "planned_mib": round(int(split[4]) / 2**20) if split else None, "transfers": split[6] if split else None,
        "measured_mib": round(int(measured[1]) / 2**20) if measured else None, "rss_mib": round(rss / 2**20) if rss else None}
    with open(f"{OUT}/{results}", "a") as out: out.write(json.dumps(row) + "\n")
    print(tag, modes, label, backend, outcome, "its", row["its"], "wall", row["wall_s"], "setup", row["setup_s"], "iter", row["iter_s"], "ms/it", row["ms_it"],
          "gpu", row["gpu_rows"], "cpu", row["cpu_rows"], "fixed", row["fixed_mib"], "planned", row["planned_mib"], "meas", row["measured_mib"], row["transfers"], flush=True)
    for path in (f06, f06.replace(".f06", ".op2")):
        if os.path.exists(path): os.remove(path)
    return row
