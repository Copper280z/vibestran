import os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
BIN = f"{OUT}/vibestran-variants"
VARS = ["both", "uniform", "interior", "allshell", "alledges"]
def study(tag, deck):
    run(tag, deck, "rigid", BIN)
    for v in VARS: run(tag, deck, "drilling", BIN, v)
    os.remove(deck)
n = 400
r = random.Random(11)
scatter = {(i, j) for i in range(n) for j in range(n) if r.random() < .05}
study("checker8", grid_sheet("checker8", n, lambda i, j: (i // 8 + j // 8) % 2 == 0))
study("checker32", grid_sheet("checker32", n, lambda i, j: (i // 32 + j // 32) % 2 == 0))
study("scatter5", grid_sheet("scatter5", n, lambda i, j: (i, j) not in scatter))
study("half", grid_sheet("half", n, lambda i, j: i < n // 2))
study("merged", unstructured_merged("merged", 410))
print("finished", flush=True)
