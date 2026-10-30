#!/bin/bash
# Smoke test of the configs (`--smoke`: 20 steps, an evaluation every 10): per neuron
# model, one H100 runs the 1,024-neuron teacher, one student and the control. From a
# snapshot of the repo; the runs go to runs/smoke/.
# Usage: bash submit_smoke.sh [models, default: pospischil hh]
set -euo pipefail
set -x
MODELS="${*:-pospischil hh}"
EXP=/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/complex-neurons
S=$(/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools/snapshot.sh)
C=$S/examples/activitome-connectome/complex-neurons/configs
RUN="~/myvenv/bin/python -P -m jaxley_pytorch.training --smoke --output-root $EXP/runs/smoke --config"
mkdir -p $EXP/jobs/smoke $EXP/runs/smoke

for m in $MODELS; do
  bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 2:00 -J cn_smoke_${m} -o $EXP/jobs/smoke/smoke_${m}_%J.log \
    "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
     $RUN $C/${m}_n1024_teacher.toml && $RUN $C/${m}_n1024_students.toml --seed 1 && \
     $RUN $C/${m}_n1024_control.toml"
done
