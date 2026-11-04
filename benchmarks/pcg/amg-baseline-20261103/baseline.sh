#!/bin/bash
# The Priority 0 baseline. Current code at every level of every class, verified at
# the coarsest against a refined direct solve; the controls from the commit before
# the smoother bound and the planner, at the coarsest two levels. Failures and
# timeouts are recorded outcomes. Any required check that fails sets status=1.
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/libcudss/12 OMP_NUM_THREADS=8
D=~/pcg-perf/baseline-20261103
SRC=$(cat $D/head.txt)
PREFIX="systemd-run --user --scope -q -p MemoryMax=26G -p MemorySwapMax=0 taskset -c 0-7"
cd ~/vibestran-pcg-perf/tools
RUN="python3 amg_model_suite.py run --work /tmp/amg-base"
STRUCT="solid-hex solid-tet10 solid-contrast solid-thin shell-thin shell-thick shell-membrane shell-cylinder-cd shell-foundation stiffened-panel space-frame rbe-assembly"
status=0
$RUN --prefix "$PREFIX" --binary $D/vibestran --results $D/current.jsonl --modes rigid,constant --direct --require-solved rigid --timeout 150 --archive $D/archive --manifest $D/manifest.json --source-commit $SRC --source-patch $D/working-tree.patch --build-dir ~/vibestran-pcg-perf/build --classes $(echo $STRUCT | tr ' ' ,) --deck rover-complete=/tmp/rover-complete.bdf --deck rover-four=/tmp/rover-four.bdf || status=1
$RUN --prefix "$PREFIX" --binary $D/vibestran --results $D/current.jsonl --modes default --timeout 300 --archive $D/archive --classes thermal-hex,magnetic-motor,magnetic-coil || status=1
$RUN --prefix "$PREFIX" --binary $D/vibestran --results $D/current.jsonl --modes constant,rigid --timeout 150 --archive $D/archive --classes shell-membrane --levels 512 || status=1
for c in $STRUCT thermal-hex magnetic-motor magnetic-coil; do
  L=$(python3 -c "import amg_model_suite as s; print(','.join(map(str, s.CLASSES['$c'][1][:2])))")
  case $c in thermal-hex|magnetic-*) M=default; T=300;; *) M=rigid,constant; T=120;; esac
  $RUN --prefix "$PREFIX" --binary $D/vibestran_control --results $D/control.jsonl --modes $M --timeout $T --archive $D/archive-control --classes $c --levels $L || status=1
done
$RUN --prefix "$PREFIX" --binary $D/vibestran_control --results $D/control.jsonl --modes rigid,constant --timeout 120 --archive $D/archive-control --deck rover-complete=/tmp/rover-complete.bdf --deck rover-four=/tmp/rover-four.bdf || status=1
$RUN --prefix "$PREFIX" --binary $D/vibestran_control --results $D/control.jsonl --modes constant,rigid --timeout 150 --archive $D/archive-control --classes shell-membrane --levels 512 || status=1
rm -f /tmp/rover-complete.bdf /tmp/rover-four.bdf
echo "finished status=$status"
