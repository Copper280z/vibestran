#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/libcudss/12 OMP_NUM_THREADS=8
D=~/pcg-perf/budget-20261104
cd ~/vibestran-pcg-perf/tools
PREFIX="systemd-run --user --scope -q -p MemoryMax=26G -p MemorySwapMax=0 taskset -c 0-7"
RUN="python3 amg_budget_study.py paired --binary $D/vibestran --work /tmp/amg-budget --results $D/paired.jsonl --budgets $D/budgets.json --archive $D/archive --timeout 300"
status=0
$RUN --prefix "$PREFIX" --pcg-benchmark $D/pcg_benchmark --repeats 5 --cells solid-hex:both-resident,solid-hex:rigid-50,solid-tet10:both-resident,solid-tet10:rigid-50,shell-membrane:both-resident,shell-membrane:rigid-50,space-frame:both-resident,space-frame:rigid-50 || status=1
$RUN --prefix "$PREFIX" --repeats 5 --cells solid-hex:rigid-host,solid-tet10:rigid-host,shell-membrane:rigid-90,space-frame:rigid-90,rbe-assembly:rigid-50 || status=1
$RUN --prefix "$PREFIX" --repeats 3 --cells rbe-assembly:rigid-75 || status=1
$RUN --prefix "$PREFIX" --repeats 2 --cells shell-thin:both-resident || status=1
echo "finished status=$status"
rm -rf /tmp/amg-budget
