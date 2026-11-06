# The final binary: thread scaling on one deck, then each backend on the decks near 240k equations.
import os, run as R
R.RES = "final.jsonl"
os.environ.pop("CYCLE_KEEP_LOGS", None)
deck = R.deck_for("solid-hex", 40)
for t in (1, 2, 4, 8, 16):
    R.run("solid-hex-40", deck, {}, threads=t, timeout=300, cores=f"0-{t - 1}", label=f"threads-{t}")
decks = [R.deck_for(*d) for d in [("shell-membrane", 200), ("shell-thin", 200), ("solid-hex", 26), ("thermal-hex", 64), ("stiffened-panel", 144), ("rbe-assembly", 128)]] + [os.path.expanduser("~/pcg-perf/drilling-20261105/rover-static.bdf")]
for deck in decks:
    tag = os.path.basename(deck)[:-4]
    for backend in ("cpu-pcg", "cuda-pcg", "cpu"):
        R.run(tag, deck, {}, threads=8, timeout=300, backend=backend)
