export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/libcudss/12 OMP_NUM_THREADS=8
for spec in "shell-membrane-400 drilling 0 stored" "shell-membrane-400 drilling 1028 computed" "solid-hex-40 rigid 0 stored" "solid-hex-40 rigid 1868 computed"; do
  set -- $spec
  if [ "$3" != 0 ]; then export VIBESTRAN_PCG_DEVICE_MEMORY_MIB=$3; else unset VIBESTRAN_PCG_DEVICE_MEMORY_MIB; fi
  out=prof-$1-$4
  nsys profile --cuda-graph-trace=node --trace=cuda --sample=none --cpuctxsw=none -o /tmp/implicit/$out --force-overwrite=true taskset -c 0-7 ./vibestran --backend=cuda-pcg --amg-modes=$2 --no-quality-checks --log-file=$out.log /tmp/implicit/$1.bdf /tmp/implicit/$out.f06 > /dev/null 2>&1
  nsys stats -r cuda_gpu_kern_sum --format csv /tmp/implicit/$out.nsys-rep 2>/dev/null > $out.csv
  grep -E "setup=|measured=" $out.log | tail -2 | cut -c1-170
  rm -f /tmp/implicit/$out.f06
done
