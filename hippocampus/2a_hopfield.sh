#!/bin/bash

# This sample script runs a Hopfield network simulation. Additional flags can
# be provided at the command line and are appended to the end of the command.

if [[ $# -lt 1 ]]; then
  echo "Usage: ./2a_hopfield.sh fileroot [-flag value ...]"
  exit 1
fi

src/search.o \
  -fileroot "out/$1" \
  -X_dir "out/pathways/N2048-p3-s512-a02-2" \
  -p 3 \
  -gamma 0.05 \
  -incomp 0. -inacc 0.01 -beta 100. \
  -n_cue 30 -T_sim 10 -T_rec 10 \
  -save_activity \
  "${@:2}"
