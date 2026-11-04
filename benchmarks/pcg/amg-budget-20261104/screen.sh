#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/libcudss/12 LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/libcudss/12 OMP_NUM_THREADS=8
D=~/pcg-perf/budget-20261104; B=~/vibestran-pcg-perf/build
cd $B && ninja vibestran pcg_benchmark 2>&1 | grep -E "error|FAILED" | head -5
cp $B/vibestran $D/vibestran; cp $B/pcg_benchmark $D/pcg_benchmark
cd ~/vibestran-pcg-perf/tools
PREFIX="systemd-run --user --scope -q -p MemoryMax=26G -p MemorySwapMax=0 taskset -c 0-7"
python3 amg_budget_study.py screen --binary $D/vibestran --work /tmp/amg-budget --results $D/screen.jsonl --budgets $D/budgets.json --archive $D/archive --prefix "$PREFIX" --timeout 300 --manifest $D/manifest.json --source-commit $(cat $D/head.txt) --source-patch $D/working-tree.patch --build-dir $B
echo "finished status=$?"
rm -rf /tmp/amg-budget
