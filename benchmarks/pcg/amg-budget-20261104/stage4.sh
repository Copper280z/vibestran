#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/libcudss/12 OMP_NUM_THREADS=8
D=~/pcg-perf/budget-20261104
cd ~/vibestran-pcg-perf/tools
PREFIX="systemd-run --user --scope -q -p MemoryMax=26G -p MemorySwapMax=0 taskset -c 0-7"
B=cuda-pcg-mixed
status=0
mkdir -p $D/$B/first-attempt; mv $D/$B/screen.jsonl $D/$B/paired.jsonl $D/$B/first-attempt/ 2>/dev/null
python3 amg_budget_study.py screen --binary $D/vibestran --work /tmp/amg-budget --results $D/$B/screen.jsonl --budgets $D/$B/budgets.json --archive $D/$B/archive --backend $B --prefix "$PREFIX" --timeout 300 --classes solid-hex,shell-thin,shell-membrane,rbe-assembly || status=1
python3 amg_budget_study.py paired --binary $D/vibestran --work /tmp/amg-budget --results $D/$B/paired.jsonl --budgets $D/$B/budgets.json --archive $D/$B/archive --backend $B --prefix "$PREFIX" --timeout 300 --repeats 5 --cells solid-hex:both-resident,solid-hex:rigid-50,shell-membrane:both-resident,shell-membrane:rigid-50 || status=1
echo "finished status=$status"
rm -rf /tmp/amg-budget
