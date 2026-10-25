#!/bin/bash
# Smoke test: smoke.sh for every network size (one H100 each), then the summary of these
# runs (one CPU job), all from a snapshot of the repo. Usage: bash submit_smoke.sh
set -euo pipefail
set -x
EXP=/nrs/scicompsoft/orhane/probes/jaxley_pytorch/activitome-connectome/network-size
S=$(/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools/snapshot.sh)
D=$S/examples/activitome-connectome/network-size
mkdir -p $EXP/jobs/smoke $EXP/runs/smoke

N512=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 1:00 -J ns_smoke_n512 -o $EXP/jobs/smoke/smoke_n512_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   bash $D/smoke.sh 512 $EXP/runs/smoke" | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
N1024=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 2:00 -J ns_smoke_n1024 -o $EXP/jobs/smoke/smoke_n1024_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   bash $D/smoke.sh 1024 $EXP/runs/smoke" | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
N2048=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 4:00 -J ns_smoke_n2048 -o $EXP/jobs/smoke/smoke_n2048_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   bash $D/smoke.sh 2048 $EXP/runs/smoke" | sed -n 's/.*<\([0-9]*\)>.*/\1/p')

bsub -P miaai -n 2 -W 0:30 -J ns_smoke_summary -w "ended($N512) && ended($N1024) && ended($N2048)" \
  -o $EXP/jobs/smoke/summary_%J.log \
  "set -o pipefail; ~/myvenv/bin/python -I $D/summarize.py \
     --dirs $EXP/runs/smoke/n512 $EXP/runs/smoke/n1024 $EXP/runs/smoke/n2048 --out $EXP/runs/smoke/summary"
