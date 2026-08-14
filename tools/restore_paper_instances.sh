#!/bin/bash
# Recover the submitted-era instance set into the repo as a first-class, additive
# dataset.  Nothing existing is overwritten: the current maps and scenarios stay
# where they are, so golden regression and in-flight experiments are unaffected.
#
#   data/maps/cross_3030_paper.map                 Square 1 of Table 2 (187x187)
#   data/scenarios/<name>-paper/<name>-paper_s<k>.scen   s0..s9, starts on the
#       aisle graph, goals at workstation sinks -- exactly as submitted.
set -e
cd ~/lima-dev

git show 1bf5a35:data/maps/cross_3030.map > data/maps/cross_3030_paper.map

declare -A SRC=(
  [warehouse-10-20-paper]=warehouse-10-20
  [warehouse-20-40-paper]=warehouse-20-40
  [cross-30-30-paper]=cross-30-30
)
for dst in "${!SRC[@]}"; do
  src="${SRC[$dst]}"
  mkdir -p "data/scenarios/${dst}"
  for s in 0 1 2 3 4 5 6 7 8 9; do
    git show "1bf5a35^:assets/${src}/scen/${src}_s${s}.scen" > "data/scenarios/${dst}/${dst}_s${s}.scen"
  done
done

echo "--- recovered ---"
ls data/maps/cross_3030_paper.map
for d in data/scenarios/*-paper; do
  printf "%-42s %s files, s0 has %s lines\n" "$d" "$(ls "$d" | wc -l)" "$(wc -l < "$d"/*_s0.scen)"
done
