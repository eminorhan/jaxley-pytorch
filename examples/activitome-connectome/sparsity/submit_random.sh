#!/bin/bash
# Random-start students (2026-10-30): per size, ten students (seeds 1-10, a job array) of the trained teacher, from
# random gains and biases drawn like the teacher's initial ones (n<N>_students_random.toml ->
# runs/n<N>_students_random/seed<k>), one RTX PRO 6000 each. The teachers, the shuffled students, the controls and
# the summary are not touched. From a snapshot of the repo. A run that stopped early continues with the same command
# plus --resume.
# Usage: bash submit_random.sh [sizes, default: 600 900]
set -euo pipefail
set -x
SIZES="${*:-600 900}"
EXP=/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/sparsity
S=$(/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools/snapshot.sh)
C=$S/examples/activitome-connectome/sparsity/configs
ENV="set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d);"
RUN="~/myvenv/bin/python -P -m jaxley_pytorch.training --output-root $EXP/runs --config"
for n in $SIZES; do
  [ -f $EXP/runs/n${n}_teacher/seed0/network.pt ] || { echo "no trained teacher for n$n" >&2; exit 2; }
  bsub -P miaai -q gpu_rtx6000 -gpu "num=1" -n 12 -W 1:00 -J "sp_n${n}_students_random[1-10]" \
    -o $EXP/jobs/n${n}_students_random_%J_%I.log "$ENV $RUN $C/n${n}_students_random.toml --seed \$LSB_JOBINDEX"
done
