# Defaults against forced computed on the largest of each class, then budgets, rover and other backends.
from base import *
OLD = os.path.expanduser("~/pcg-perf/drilling-20261105/vibestran")
for name, (_, levels, _k) in suite.CLASSES.items():
    size = levels[-1]
    deck = f"{TMP}/{name}-{size}.bdf"
    if not os.path.exists(deck): suite.write_deck(name, size, deck)
    tag = f"{name}-{size}"
    stored = run(tag, deck, None, 0, "default")
    if stored["planned_mib"] and stored["transfers"] == "stored": run(tag, deck, None, stored["planned_mib"] - 2, "computed")
    os.remove(deck)
for name, size in (("shell-membrane", 1258), ("solid-hex", 70)):
    deck = f"{TMP}/{name}-{size}.bdf"; suite.write_deck(name, size, deck)
    tag = f"{name}-{size}"
    run(tag, deck, None, 0, "default")
    log = open(f"{OUT}/{tag}-None-default-cuda-pcg.log", errors="replace").read()
    fr = re.search(r"row budgets: 50%=(\d+) 75%=(\d+) 90%=(\d+)", log) or re.search(r"row budgets: 50%=(\d+).*75%=(\d+).*90%=(\d+)", log)
    if fr:
        for f, v in zip((50, 75, 90), fr.groups()):
            run(tag, deck, None, int(v) // 2**20, f"rows{f}")
    run(tag, deck, "drilling", 0, "before", binary=OLD)
    log = open(f"{OUT}/{tag}-drilling-before-cuda-pcg.log", errors="replace").read()
    fr = re.search(r"row budgets: 50%=(\d+) 75%=(\d+) 90%=(\d+)", log) or re.search(r"row budgets: 50%=(\d+).*75%=(\d+).*90%=(\d+)", log)
    if fr:
        for f, v in zip((50, 75, 90), fr.groups()):
            run(tag, deck, "drilling", int(v) // 2**20, f"before-rows{f}", binary=OLD)
    os.remove(deck)
for tag in ("rover-complete", "rover-static"):
    deck = os.path.expanduser(f"~/pcg-perf/drilling-20261105/{tag}.bdf")
    stored = run(tag, deck, None, 0, "default")
    if stored["planned_mib"]: run(tag, deck, None, stored["planned_mib"] - 2, "computed")
deck = f"{TMP}/membrane-200.bdf"; suite.write_deck("shell-membrane", 200, deck)
for backend in ("cuda-pcg-mixed", "vulkan"):
    stored = run("membrane-200", deck, None, 0, "default", backend=backend)
    if stored["planned_mib"]: run("membrane-200", deck, None, stored["planned_mib"] - 2, "computed", backend=backend)
    if stored["planned_mib"]: run("membrane-200", deck, None, stored["planned_mib"] * 45 // 100, "spill", backend=backend)
os.remove(deck)
deck = f"{TMP}/membrane-400.bdf"; suite.write_deck("shell-membrane", 400, deck)
stored = run("membrane-400", deck, None, 0, "default", backend="vulkan")
if stored["planned_mib"]: run("membrane-400", deck, None, stored["planned_mib"] - 2, "computed", backend="vulkan")
deck2 = f"{TMP}/hex-40.bdf"; suite.write_deck("solid-hex", 40, deck2)
stored = run("hex-40", deck2, None, 0, "default", backend="vulkan")
if stored["planned_mib"]: run("hex-40", deck2, None, stored["planned_mib"] - 2, "computed", backend="vulkan")
os.remove(deck); os.remove(deck2)
print("finished", flush=True)
