#!/bin/bash
# Kills any vibestran whose resident memory passes the limit in GiB, and logs the peak.
limit=$((${1:-28} * 1048576)); peak=0
while sleep 0.5; do
  for p in $(pgrep -x vibestran); do
    r=$(awk "/VmRSS/{print \$2}" /proc/$p/status 2>/dev/null); r=${r:-0}
    [ "$r" -gt "$peak" ] && peak=$r
    if [ "$r" -gt "$limit" ]; then echo "killed $p at $r KiB"; kill -9 $p; fi
  done
done
