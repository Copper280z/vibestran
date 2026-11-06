export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/libcudss/12 OMP_NUM_THREADS=8
T=~/vibestran-pcg-perf/tools/amg_model_suite.py; O=~/pcg-perf/implicit-20261105
python3 $T run --binary $O/vibestran --work /tmp/implicit/suite --results $O/suite.json --backend cuda-pcg --modes default --no-probe --direct --prefix "taskset -c 0-7" --timeout 900
rm -rf /tmp/implicit/suite
python3 $T run --binary $O/vibestran --work /tmp/implicit/suite --results $O/suite-budget.json --backend cuda-pcg --modes default --no-probe --direct --prefix "taskset -c 0-7" --timeout 600 --budget-mib 200 --classes solid-hex,solid-tet10,shell-membrane,shell-thin,rbe-assembly
rm -rf /tmp/implicit/suite
echo done
