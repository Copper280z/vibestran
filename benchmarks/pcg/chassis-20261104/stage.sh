#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/libcudss/12 LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/libcudss/12 OMP_NUM_THREADS=8
D=~/pcg-perf/chassis-20261104
PREFIX="systemd-run --user --scope -q -p MemoryMax=26G -p MemorySwapMax=0 taskset -c 0-7"
python3 $D/variants.py > $D/variants.log 2>&1
echo variants-done >> $D/stage.log
cd ~/vibestran-pcg-perf/tools
status=0
python3 amg_budget_study.py screen --binary $D/vibestran --work /tmp/amg-budget --results $D/screen.jsonl --budgets $D/budgets.json --archive $D/archive --prefix "$PREFIX" --timeout 300 --classes tube-chassis --manifest $D/manifest.json --source-commit $(cat $D/head.txt) --source-patch $D/working-tree.patch --build-dir ~/vibestran-pcg-perf/build >> $D/stage.log 2>&1 || status=1
for B in vulkan cuda-pcg-mixed; do
  mkdir -p $D/$B
  python3 amg_budget_study.py screen --binary $D/vibestran --work /tmp/amg-budget --results $D/$B/screen.jsonl --budgets $D/$B/budgets.json --archive $D/$B/archive --backend $B --prefix "$PREFIX" --timeout 300 --classes tube-chassis >> $D/stage.log 2>&1 || status=1
done
rm -rf /tmp/amg-budget
echo "finished status=$status" >> $D/stage.log
