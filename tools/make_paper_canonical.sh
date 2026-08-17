#!/bin/bash
# Make the submitted-era instance set canonical.
#
# The generated instances are kept under *_gen / *-gen names rather than deleted,
# so every measurement taken on them stays reproducible and the two can be
# compared later.  After this runs, the plain names refer to exactly what the
# submitted version of the paper used:
#   data/maps/{warehouse_10_20,warehouse_20_40,cross_3030}.map
#   data/scenarios/{warehouse-10-20,warehouse-20-40,cross-30-30}/*_s0..s9.scen
set -e
cd ~/lima-dev

# 1. maps: park the generated geometry, install the submitted-era files
for m in warehouse_10_20 warehouse_20_40 cross_3030; do
  [ -f "data/maps/${m}_gen.map" ] || cp "data/maps/${m}.map" "data/maps/${m}_gen.map"
done
git show '1bf5a35^:assets/warehouse-10-20/warehouse-10-20.map' > data/maps/warehouse_10_20.map
git show '1bf5a35^:assets/warehouse-20-40/warehouse-20-40.map' > data/maps/warehouse_20_40.map
git show '1bf5a35:data/maps/cross_3030.map'                    > data/maps/cross_3030.map

# 2. scenarios: park the generated rollouts, install the submitted-era ones
declare -A SRC=(
  [warehouse-10-20]=warehouse-10-20
  [warehouse-20-40]=warehouse-20-40
  [cross-30-30]=cross-30-30
)
for name in "${!SRC[@]}"; do
  if [ -d "data/scenarios/${name}" ] && [ ! -d "data/scenarios/${name}-gen" ]; then
    mkdir -p "data/scenarios/${name}-gen"
    for f in "data/scenarios/${name}"/*.scen; do
      base=$(basename "$f")
      mv "$f" "data/scenarios/${name}-gen/${base/${name}_/${name}-gen_}"
    done
  fi
  mkdir -p "data/scenarios/${name}"
  for s in 0 1 2 3 4 5 6 7 8 9; do
    git show "1bf5a35^:assets/${SRC[$name]}/scen/${SRC[$name]}_s${s}.scen" \
      > "data/scenarios/${name}/${name}_s${s}.scen"
  done
done

# 3. the transitional *-paper copies are now redundant
rm -rf data/scenarios/warehouse-10-20-paper data/scenarios/warehouse-20-40-paper \
       data/scenarios/cross-30-30-paper data/maps/cross_3030_paper.map

echo "--- canonical instance set ---"
for m in warehouse_10_20 warehouse_20_40 cross_3030; do
  printf "%-22s %s\n" "$m" "$(head -3 data/maps/${m}.map | tr '\n' ' ')"
done
for d in data/scenarios/*/; do
  printf "%-40s %2s files\n" "$d" "$(ls "$d" | wc -l)"
done
