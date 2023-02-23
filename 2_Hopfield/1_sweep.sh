#!/bin/bash

../../SharedC/search.o \
  -fileroot "out/$1" -threads 1 \
  -X_dir "../models/N2048-p3-s512-a02-2" \
  -p 3 \
  -gamma 0.05 \
  -incomp 0. -inacc 0.01 -beta 100. \
  -n_cue 30 -T_sim 10 -T_rec 10 \
  -no_search -no_shuffle -save_activity \
  "${@:2}"
