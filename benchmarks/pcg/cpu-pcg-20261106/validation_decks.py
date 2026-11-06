import os, re, subprocess, sys, json, tempfile, collections, concurrent.futures as cf
S = sys.argv[1]; ROOT = "/Users/bob/Documents/vibe-tran/MYSTRAN_Validation/decks"
BINS = {"direct": ("/Users/bob/Documents/vibe-tran/build/vibestran", "--backend=cpu"), "amg": ("/Users/bob/Documents/vibe-tran/build/vibestran", "--backend=cpu-pcg"), "ic": (S + "/wt-main/build/vibestran", "--backend=cpu-pcg")}
decks = []
for d, _, files in os.walk(ROOT):
    for f in files:
        if f.lower().endswith((".bdf", ".dat")): decks.append(os.path.join(d, f))
def run(args):
    deck, name = args
    b, flag = BINS[name]
    with tempfile.TemporaryDirectory() as t:
        env = {**os.environ, "OMP_NUM_THREADS": "1"}
        try:
            p = subprocess.run([b, flag, deck, t + "/o.f06"], capture_output=True, text=True, timeout=30, env=env, errors="replace")
            text = p.stdout + p.stderr
            m = re.findall(r"(?:Solver error|error\]?)[: ]+(.*)", text)
            return deck, name, p.returncode, (m[-1][:110] if m else ""), os.path.exists(t + "/o.f06")
        except subprocess.TimeoutExpired:
            return deck, name, "timeout", "", False
out = collections.defaultdict(dict)
with cf.ThreadPoolExecutor(6) as ex:
    for deck, name, rc, msg, f06 in ex.map(run, [(d, n) for d in decks for n in BINS]):
        out[os.path.relpath(deck, ROOT)][name] = (rc, msg)
json.dump(out, open(S + "/decks.json", "w"), indent=0)
print("decks", len(out))
c = collections.Counter()
for d, r in out.items():
    c[tuple((n, r[n][0] == 0) for n in ("direct", "ic", "amg"))] += 1
for k, v in c.most_common(): print(v, k)
msgs = collections.Counter(re.sub(r"[-\d.e+]+", "#", r["amg"][1]) for r in out.values() if r["direct"][0] == 0 and r["amg"][0] != 0)
print("amg fails where direct ok:"); [print(" ", v, k) for k, v in msgs.most_common(12)]
msgs = collections.Counter(re.sub(r"[-\d.e+]+", "#", r["ic"][1]) for r in out.values() if r["direct"][0] == 0 and r["ic"][0] != 0)
print("ic fails where direct ok:"); [print(" ", v, k) for k, v in msgs.most_common(12)]
