#!/bin/bash

if [[ $# -lt 1 ]]; then
  echo "Usage: ./sweep_batch.sh s1 [s2 ...]"
  exit 1
fi

ss=${@:1}
cue_types=( "sparse" "dense" "both" )
target_types=( "sparse" "category" )

args=()

for cue in "${cue_types[@]}"; do
  for target in "${target_types[@]}"; do

    root="${cue:0:1}${target:0:1}"

    case "$target" in
      "sparse"   ) theta=0.50 ;;
      "category" ) theta=0.10 ;;
    esac

  for s in ${ss[@]}; do

      arg="./sim.sh $root-s$s -s $s"
      arg+=" -${cue}_cue -${target}_target -theta $theta"
      arg+=" -no_search -no_shuffle"
      args+=("$arg")

    done

  done
done

if [[ $(uname -s) == "Darwin" ]]
then
  ncores=$(sysctl -n hw.physicalcpu)
else
  ncores=$(nproc)
fi

parallel --dry-run ::: "${args[@]}"
echo -n "run simulations (y/n)? "
read a;
if [[ "$a" == "y" ]]; then
  parallel --ungroup -j "$ncores" ::: "${args[@]}"
  echo "View overlaps with \"./overlaps.sh\""
fi

