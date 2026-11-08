#!/bin/bash
# Per number of cell types (4, 8, 16; default all): the teacher (configs/t<K>_teacher.toml, 5,000 steps, as
# complex-neurons' Pospischil teacher), then, when it is done, its ten students (seeds 1-10, a job array, start
# shuffled among the types of each class) and the control (the teacher's own start), 2,000 steps each. One GPU each:
# an RTX PRO 6000 (QUEUE=gpu_h100 for an H100). From a snapshot of the repo; the runs go to
# runs/<experiment_name>/seed<k>; a run that stopped early continues with the same command plus --resume. Time
# limits as complex-neurons' (Pospischil about 14 s per step): teacher 30 h, students and control 14 h. When the
# students and controls have ended, the summary (../complex-neurons/summarize.py, one group per number of types and
# complex-neurons' Pospischil runs as the per-neuron reference) writes the tables and figures to figures/.
# Usage: [QUEUE=gpu_h100] bash submit.sh [4] [8] [16]
set -euo pipefail
set -x
TYPES="${*:-4 8 16}"
QUEUE="${QUEUE:-gpu_rtx6000}"
case "$QUEUE" in gpu_h100|gpu_rtx6000) ;; *) echo "QUEUE must be gpu_h100 or gpu_rtx6000" >&2; exit 2 ;; esac
EXP=/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/cell-types
S=$(/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools/snapshot.sh)
C=$S/examples/activitome-connectome/cell-types/configs
ENV="set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d);"
RUN="~/myvenv/bin/python -P -m jaxley_pytorch.training --output-root $EXP/runs --config"
CN=/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/complex-neurons
mkdir -p $EXP/runs $EXP/jobs $EXP/figures
ENDED=()
GROUP_ARGS=()
for K in $TYPES; do
  case "$K" in 4|8|16) ;; *) echo "no configs for $K cell types" >&2; exit 2 ;; esac
  T=$(bsub -P miaai -q $QUEUE -gpu "num=1" -n 12 -W 30:00 -J ct_t${K}_teacher \
    -o $EXP/jobs/t${K}_teacher_%J.log "$ENV $RUN $C/t${K}_teacher.toml" | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
  STU=$(bsub -P miaai -q $QUEUE -gpu "num=1" -n 12 -W 14:00 -w "done($T)" -J "ct_t${K}_students[1-10]" \
    -o $EXP/jobs/t${K}_students_%J_%I.log "$ENV $RUN $C/t${K}_students.toml --seed \$LSB_JOBINDEX" \
    | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
  CON=$(bsub -P miaai -q $QUEUE -gpu "num=1" -n 12 -W 14:00 -w "done($T)" -J ct_t${K}_control \
    -o $EXP/jobs/t${K}_control_%J.log "$ENV $RUN $C/t${K}_control.toml" | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
  ENDED+=("ended($STU)" "ended($CON)")
  GROUP_ARGS+=(--group "\"$K types\"" $EXP/runs t$K)
done

# Summary: every number of types submitted, and the per-neuron reference.
DEP=$(printf ' && %s' "${ENDED[@]}")
bsub -P miaai -n 2 -W 0:30 -J ct_summary -w "${DEP:4}" -o $EXP/jobs/summary_%J.log \
  "set -o pipefail; export PYTHONPATH=$S; ~/myvenv/bin/python -P $S/examples/activitome-connectome/complex-neurons/summarize.py \
     ${GROUP_ARGS[*]} --group \"1,024 types (complex-neurons)\" $CN/runs pospischil_n1024 --out $EXP/figures"
