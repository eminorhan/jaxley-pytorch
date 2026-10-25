#!/bin/bash
# Stage 0: the BPE token files (one CPU job), then the smoke test (one H100, after the
# token job), both from a snapshot of the repo. Usage: bash submit_smoke.sh
set -euo pipefail
set -x
EXP=/nrs/scicompsoft/orhane/probes/jaxley_pytorch/activitome-connectome/task-complexity
S=$(/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools/snapshot.sh)
D=$S/examples/activitome-connectome/task-complexity
mkdir -p $EXP/data $EXP/jobs/smoke $EXP/runs/smoke

TOKENS=$(bsub -P miaai -n 16 -W 2:00 -J tc_tokens -o $EXP/jobs/prepare_tokens_%J.log \
  "set -o pipefail; export RAYON_NUM_THREADS=16 OMP_NUM_THREADS=16; \
   ~/myvenv/bin/python -I $D/prepare_tokens.py \
     --parquet /groups/scicompsoft/home/orhane/huggingface/hub/datasets--karpathy--tinystories-gpt4-clean/snapshots/0397e27157956705a0260709da3095bb9c43d6a7/tinystories_gpt4_clean.parquet \
     --vocab 4096 --out $EXP/data/bpe4096" | sed -n 's/.*<\([0-9]*\)>.*/\1/p')

bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 1:00 -J tc_smoke -w "done($TOKENS)" \
  -o $EXP/jobs/smoke/smoke_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   bash $D/smoke.sh $EXP/runs/smoke"
