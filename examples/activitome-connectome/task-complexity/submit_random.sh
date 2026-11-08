#!/bin/bash
# Random-start students of the text tasks (2026-10-30; the cycling task has them already, submit.sh): per task, ten
# students (seeds 1-10, a job array) of the trained teacher, from random gains and biases drawn like the teacher's
# initial ones (<task>_students_random.toml -> runs/<task>_students_random/seed<k>), one RTX PRO 6000 each. The
# teachers, the shuffled students, the controls and the figures are not touched. From a snapshot of the repo.
# Usage: bash submit_random.sh [tasks, default: chars tokens]
set -euo pipefail
set -x
TASKS="${*:-chars tokens}"
EXP=/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/task-complexity
S=$(/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools/snapshot.sh)
C=$S/examples/activitome-connectome/task-complexity/configs
ENV="set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d);"
RUN="~/myvenv/bin/python -P -m jaxley_pytorch.training --output-root $EXP/runs --config"
for t in $TASKS; do
  # (these teachers predate network.pt: their checkpoint is teacher.pt, which read_network also reads)
  [ -f $EXP/runs/${t}_teacher/seed0/network.pt ] || [ -f $EXP/runs/${t}_teacher/seed0/teacher.pt ] \
    || { echo "no trained teacher for $t" >&2; exit 2; }
  bsub -P miaai -q gpu_rtx6000 -gpu "num=1" -n 12 -W 4:00 -J "tc_${t}_students_random[1-10]" \
    -o $EXP/jobs/${t}_students_random_%J_%I.log "$ENV $RUN $C/${t}_students_random.toml --seed \$LSB_JOBINDEX"
done
