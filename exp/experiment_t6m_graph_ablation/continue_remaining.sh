#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
while [ "$(find results/G1_BIOMARKER_SELF_ONLY -mindepth 2 -maxdepth 2 -name metrics.json 2>/dev/null | wc -l)" -lt 10 ]; do sleep 30; done
for group in G2_BIOMARKER_Q97 G3_BIOMARKER_Q95 G4_NO_PEN_PHYSICAL G5_NO_PEN_NO_GFR_PHYSICAL; do
  PYTHONDONTWRITEBYTECODE=1 python run.py "$group" --parallel --folds-to-run 10
 done
