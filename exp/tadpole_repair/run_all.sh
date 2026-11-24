#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

if (( $# )); then
    variants=("$@")
else
    variants=(previous_control paper_config clip_xavier tadpole_prompts mmgpl_concepts
              mmgpl_class_prompts mmgpl_class_graph_sup mmgpl_class_pen_graph_sup
              mmgpl_class_full_graph_sup pen_equations gfr_sum gfr_softmax graph_supervision
              graph_frobenius combined)
fi
for variant in "${variants[@]}"; do
    (
        flock -x 9
        python -u run.py "$variant" 2>&1 | tee -a "logs/$variant.log"
    ) 9>"logs/$variant.lock"
done
