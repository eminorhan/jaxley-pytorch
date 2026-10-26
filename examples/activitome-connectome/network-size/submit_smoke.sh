#!/bin/bash
# Smoke test of the configs (`--smoke`: 20 steps, an evaluation every 10): per network
# size, one H100 runs the teacher, one student and the control; then the summary of
# these runs (one CPU job). From a snapshot of the repo; the runs go to runs/smoke/.
# Usage: bash submit_smoke.sh
set -euo pipefail
set -x
EXP=/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/network-size
S=$(/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools/snapshot.sh)
D=$S/examples/activitome-connectome/network-size
mkdir -p $EXP/jobs/smoke $EXP/runs/smoke

N512=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 1:00 -J ns_smoke_n512 -o $EXP/jobs/smoke/smoke_n512_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --smoke --config $D/configs/n512_teacher.toml \
     --output-root $EXP/runs/smoke && \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --smoke --config $D/configs/n512_students.toml \
     --output-root $EXP/runs/smoke --seed 1 && \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --smoke --config $D/configs/n512_control.toml \
     --output-root $EXP/runs/smoke" | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
N1024=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 2:00 -J ns_smoke_n1024 -o $EXP/jobs/smoke/smoke_n1024_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --smoke --config $D/configs/n1024_teacher.toml \
     --output-root $EXP/runs/smoke && \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --smoke --config $D/configs/n1024_students.toml \
     --output-root $EXP/runs/smoke --seed 1 && \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --smoke --config $D/configs/n1024_control.toml \
     --output-root $EXP/runs/smoke" | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
N2048=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 4:00 -J ns_smoke_n2048 -o $EXP/jobs/smoke/smoke_n2048_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --smoke --config $D/configs/n2048_teacher.toml \
     --output-root $EXP/runs/smoke && \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --smoke --config $D/configs/n2048_students.toml \
     --output-root $EXP/runs/smoke --seed 1 && \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --smoke --config $D/configs/n2048_control.toml \
     --output-root $EXP/runs/smoke" | sed -n 's/.*<\([0-9]*\)>.*/\1/p')

bsub -P miaai -n 2 -W 0:30 -J ns_smoke_summary -w "ended($N512) && ended($N1024) && ended($N2048)" \
  -o $EXP/jobs/smoke/summary_%J.log \
  "set -o pipefail; export PYTHONPATH=$S; ~/myvenv/bin/python -P $D/summarize.py --runs $EXP/runs/smoke \
     --sizes 512 1024 2048 --out $EXP/runs/smoke/summary"
