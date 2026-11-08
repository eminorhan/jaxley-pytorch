#!/bin/bash
# Smoke test of the configs (`--smoke`: 20 steps, an evaluation every 10): per number of cell types, one RTX PRO 6000
# runs the teacher, one student and the control. From a snapshot of the repo; the runs go to runs/smoke/.
# Usage: bash submit_smoke.sh [4] [8] [16]
set -euo pipefail
set -x
TYPES="${*:-4 8 16}"
EXP=/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/cell-types
S=$(/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools/snapshot.sh)
C=$S/examples/activitome-connectome/cell-types/configs
RUN="~/myvenv/bin/python -P -m jaxley_pytorch.training --smoke --output-root $EXP/runs/smoke --config"
mkdir -p $EXP/jobs/smoke $EXP/runs/smoke
for K in $TYPES; do
  bsub -P miaai -q gpu_rtx6000 -gpu "num=1" -n 12 -W 2:00 -J ct_smoke_t${K} -o $EXP/jobs/smoke/smoke_t${K}_%J.log \
    "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
     $RUN $C/t${K}_teacher.toml && $RUN $C/t${K}_students.toml --seed 1 && $RUN $C/t${K}_control.toml"
done
