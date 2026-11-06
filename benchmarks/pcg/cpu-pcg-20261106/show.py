import json, sys
for l in open(sys.argv[1] if len(sys.argv)>1 else "runs.jsonl"):
    r=json.loads(l)
    print("%-20s %-34s t%-2s rc %-7s wall %6s setup %6s iter %7s its %5s ms/it %7s err %-9s rss %4s rpl %s"%(r["tag"],r["label"],r.get("threads"),r["rc"],r["wall"],r.get("setup_s"),r.get("iter_s"),r.get("its"),r.get("ms_it"),r.get("err"),r.get("rss_gib"),r.get("rows_per_list")))
