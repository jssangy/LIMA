#!/usr/bin/env bash
set -euo pipefail

cmake -S /home/shlee/mapf-baselines/pibt2 \
  -B /home/shlee/mapf-baselines/pibt2/build_bonus \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /home/shlee/mapf-baselines/pibt2/build_bonus \
  --target lifelong_fixed -j2

cmake -S /home/shlee/mapf-baselines/MAPF-LNS2 \
  -B /home/shlee/mapf-baselines/MAPF-LNS2/build_bonus \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /home/shlee/mapf-baselines/MAPF-LNS2/build_bonus -j2
