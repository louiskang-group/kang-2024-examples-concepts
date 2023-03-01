#!/bin/bash

if [[ $# -ne 0 ]]; then
  echo "Usage: ./$0"
  exit 1
fi

src/dynamics.o \
  -fileroot "out/oscillation/baseline" \
  -X_dir "out/pathways/N2048-p3-s512-a02-2" \
  -p 3 -s 50 \
  -gamma 0.05 \
  -square -theta_range 0.6 0.2 20 \
  -incomp 0. -inacc 0.01 -beta 100 \
  -n_cue 30 -T_sim 120 -T_rec -2 \
  -save_activity
