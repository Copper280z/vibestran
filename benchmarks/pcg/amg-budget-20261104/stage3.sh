#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/libcudss/12 OMP_NUM_THREADS=8
D=~/pcg-perf/budget-20261104
cd ~/vibestran-pcg-perf/tools
PREFIX="systemd-run --user --scope -q -p MemoryMax=26G -p MemorySwapMax=0 taskset -c 0-7"
status=0
python3 amg_budget_study.py large --binary $D/vibestran --work /tmp/amg-budget --results $D/large.jsonl --archive $D/archive/large --prefix "$PREFIX" --timeout 1200 --repeats 2 --cells solid-hex:65,solid-hex:70,shell-membrane:1258 || status=1
python3 amg_budget_study.py large --binary $D/vibestran --work /tmp/amg-budget --results $D/large.jsonl --archive $D/archive/large --prefix "$PREFIX" --timeout 1200 --repeats 1 --cells shell-membrane:1320 || status=1
echo "large finished status=$status"
for B in cuda-pcg-mixed vulkan; do
  mkdir -p $D/$B
  python3 amg_budget_study.py screen --binary $D/vibestran --work /tmp/amg-budget --results $D/$B/screen.jsonl --budgets $D/$B/budgets.json --archive $D/$B/archive --backend $B --prefix "$PREFIX" --timeout 300 --classes solid-hex,shell-thin,shell-membrane,rbe-assembly || status=1
  python3 amg_budget_study.py paired --binary $D/vibestran --work /tmp/amg-budget --results $D/$B/paired.jsonl --budgets $D/$B/budgets.json --archive $D/$B/archive --backend $B --prefix "$PREFIX" --timeout 300 --repeats 5 --cells solid-hex:both-resident,solid-hex:rigid-50,shell-membrane:both-resident,shell-membrane:rigid-50 || status=1
  echo "$B finished status=$status"
done
echo "finished status=$status"
rm -rf /tmp/amg-budget
