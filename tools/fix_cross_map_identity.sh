#!/bin/bash
# Make the plain name data/maps/cross_3030.map refer to the map the paper used.
#
# Table 2's third row ("Square 1") is reproduced exactly, under the original
# Python simulator's own definitions, only by the 187x187 cross map:
#   tiles 10200, #V 900, #E 1860.
# The 237x177 geometry that occupied this filename since commit 7b8b3b3 gives
# 17834 / 840 / 1738 and matches no row.  The frozen campaign already sources
# data/maps/cross_3030_paper.map, so this only removes the trap left in the
# working tree for anyone using the obvious filename.
#
# Nothing that is currently running reads these paths: the live jobs read
# results/revision_final/certified_inputs_v*/maps/*.  The golden regression is
# repointed to the parked file by name so its baselines stay valid.
set -e
cd ~/lima-dev

if [ ! -f data/maps/cross_3030_paper.map ]; then
  echo "missing data/maps/cross_3030_paper.map" >&2; exit 1
fi

# 1. park the 237x177 geometry under an explicit name
if [ ! -f data/maps/cross_3030_open237x177.map ]; then
  git mv data/maps/cross_3030.map data/maps/cross_3030_open237x177.map
fi

# 2. install the paper geometry under the plain name
cp data/maps/cross_3030_paper.map data/maps/cross_3030.map

# 3. the generated cross scenarios belong to the parked geometry, so park them too
if [ -d data/scenarios/cross-30-30 ] && [ ! -d data/scenarios/cross-30-30-open237 ]; then
  git mv data/scenarios/cross-30-30 data/scenarios/cross-30-30-open237
  for f in data/scenarios/cross-30-30-open237/cross-30-30_s*.scen; do
    [ -f "$f" ] || continue
    git mv "$f" "${f/cross-30-30_s/cross-30-30-open237_s}"
  done
fi

# 4. keep the golden regression testing exactly what it tested before, by name
sed -i 's|^cross_3030\.map|cross_3030_open237x177.map|; s|\|cross-30-30\||\|cross-30-30-open237\||' \
  tests/golden/e0_quick.golden

echo "--- result ---"
python3 tools/tbl2_python_rule.py data/maps/cross_3030.map data/maps/cross_3030_open237x177.map
echo
grep -c . tests/golden/e0_quick.golden | sed 's/^/golden cells: /'
grep "^cross" tests/golden/e0_quick.golden | cut -c1-60
ls -d data/scenarios/*/
