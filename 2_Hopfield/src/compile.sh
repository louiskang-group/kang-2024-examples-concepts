#!/bin/bash
if [[ $# -ne 1 ]]; then
  echo "Error: one argument required"
  exit 1
fi

gcc -o "${1/%.c/.o}" "$1" -march=native -mtune=native \
  -Ofast -lziggurat -Wno-format-overflow \
  -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -L${CMPLR_ROOT}/mac/compiler/lib -Wl,-rpath,${CMPLR_ROOT}/mac/compiler/lib \
  -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
