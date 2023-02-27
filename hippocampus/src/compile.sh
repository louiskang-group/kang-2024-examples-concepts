#!/bin/bash
if [[ $# -ne 1 ]]; then
  echo "Error: one argument required (e.g. ./compile.sh search.c)"
  exit 1
fi


args=" -DMKL_ILP64 -m64 -I\"${MKLROOT}/include\""
if [[ $(uname -s) == "Darwin" ]]
then
  args+=" -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl"
else
  args+=" -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl"
  args+=" -Wno-format-overflow"
fi

gcc -o "${1/%.c/.o}" "$1" -march=native -mtune=native -Ofast -lziggurat $args
