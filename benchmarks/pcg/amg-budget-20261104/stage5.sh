#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/libcudss/12 OMP_NUM_THREADS=8
D=~/pcg-perf/budget-20261104
cd ~/vibestran-pcg-perf/tools
PREFIX="systemd-run --user --scope -q -p MemoryMax=26G -p MemorySwapMax=0 taskset -c 0-7"
DECKS="--deck rover-complete=/tmp/amg-rover/rover-complete.bdf --deck rover-static=/tmp/amg-rover/rover-static.bdf"
COMMON="--binary $D/vibestran --work /tmp/amg-budget --budgets $D/rover/budgets.json --archive $D/rover/archive --prefix"
status=0
python3 amg_budget_study.py screen $COMMON "$PREFIX" --results $D/rover/screen.jsonl --timeout 300 $DECKS --classes rover-complete,rover-static || status=1
python3 amg_budget_study.py paired $COMMON "$PREFIX" --results $D/rover/paired.jsonl --timeout 300 $DECKS --repeats 5 --cells rover-complete:both-resident,rover-complete:rigid-50,rover-complete:rigid-host,rover-static:both-resident,rover-static:rigid-50,rover-static:rigid-host || status=1
echo "finished status=$status"
rm -rf /tmp/amg-budget /tmp/amg-rover
