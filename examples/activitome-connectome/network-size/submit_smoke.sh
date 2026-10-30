#!/bin/bash
# Smoke test of the configs (`--smoke`: 20 steps, an evaluation every 10): per network
# size, one H100 runs the teacher, one student and the control; then the summary of
# these runs (one CPU job), with task-complexity's token runs as the 300-neuron size.
# From a snapshot of the repo; the runs go to runs/smoke/.
# Usage: bash submit_smoke.sh
set -euo pipefail
set -x
EXP=/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/network-size
TC=/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/task-complexity
S=$(/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools/snapshot.sh)
D=$S/examples/activitome-connectome/network-size
mkdir -p $EXP/jobs/smoke $EXP/runs/smoke

N600=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 0:30 -J ns_smoke_n600 -o $EXP/jobs/smoke/smoke_n600_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --smoke --config $D/configs/n600_teacher.toml \
     --output-root $EXP/runs/smoke && \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --smoke --config $D/configs/n600_students.toml \
     --output-root $EXP/runs/smoke --seed 1 && \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --smoke --config $D/configs/n600_control.toml \
     --output-root $EXP/runs/smoke" | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
N900=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 0:30 -J ns_smoke_n900 -o $EXP/jobs/smoke/smoke_n900_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --smoke --config $D/configs/n900_teacher.toml \
     --output-root $EXP/runs/smoke && \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --smoke --config $D/configs/n900_students.toml \
     --output-root $EXP/runs/smoke --seed 1 && \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --smoke --config $D/configs/n900_control.toml \
     --output-root $EXP/runs/smoke" | sed -n 's/.*<\([0-9]*\)>.*/\1/p')

bsub -P miaai -n 2 -W 0:30 -J ns_smoke_summary -w "ended($N600) && ended($N900)" \
  -o $EXP/jobs/smoke/summary_%J.log \
  "set -o pipefail; export PYTHONPATH=$S; ~/myvenv/bin/python -P $D/summarize.py --runs $EXP/runs/smoke \
     --sizes 600 900 --task-complexity $TC/runs --out $EXP/runs/smoke/summary"
