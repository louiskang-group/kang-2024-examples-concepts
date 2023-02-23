#!/bin/bash

../../Code/Overlap/hopfield.o \
  -fileroot "fine/$1" -threads 1 \
  -X_dir "../../Patterns/N10000-p200-s1500-a01-5-c2" \
  -gamma 0.1 -beta 0. \
  -noiseless \
  -sparse_cue -sparse_target \
  -n_cue 20 -T_sim 10 -T_rec 2 \
  -theta_mid 0.64 -n_round 2 -n_value 7 \
  -n_search 20 -T_search 5 \
  "${@:2}"
