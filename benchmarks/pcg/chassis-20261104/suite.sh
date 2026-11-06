#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/libcudss/12 OMP_NUM_THREADS=8
cd ~/vibestran-pcg-perf/tools
python3 amg_model_suite.py run --binary /home/bob/pcg-perf/chassis-20261104/vibestran --work /tmp/amg-chassis --results /home/bob/pcg-perf/chassis-20261104/suite.jsonl --archive /home/bob/pcg-perf/chassis-20261104/suite-archive --classes tube-chassis --modes rigid,constant --direct --verify-levels 3 --prefix "systemd-run --user --scope -q -p MemoryMax=26G -p MemorySwapMax=0 taskset -c 0-7" --timeout 300 > /home/bob/pcg-perf/chassis-20261104/suite.log 2>&1
echo finished status=$? >> /home/bob/pcg-perf/chassis-20261104/suite.log
python3 amg_model_suite.py report /home/bob/pcg-perf/chassis-20261104/suite.jsonl >> /home/bob/pcg-perf/chassis-20261104/suite.log 2>&1
rm -rf /tmp/amg-chassis
