#!/bin/bash
# Stage 2 at 1,024 neurons, per model (pospischil, hh; default both): the teacher (configs/<model>_n1024_teacher.toml,
# 5,000 steps, surrogate gradient na_activation_gradient = 0.1, lr 3e-3), then, when it is done, its ten students
# (seeds 1-10, a job array, shuffled start) and the control (the teacher's own start), 2,000 steps each. One GPU each:
# an RTX PRO 6000 (QUEUE=gpu_h100 for an H100). From a snapshot of the repo; the runs go to
# runs/<experiment_name>/seed<k>; a run that stopped early continues with the same command plus --resume.
# Time limits: about 1.4x the pilots' step times on an RTX PRO 6000 (Pospischil 14.2 s, Hodgkin-Huxley 12.3 s per
# step, plus about 3 minutes per evaluation): teacher about 21 h, students and control about 9 h.
# With HOLD_TEACHERS=1 the teachers are submitted held (bsub -H; they keep their place in the queue): release them with
# bresume once their run directories are free.
# Usage: [QUEUE=gpu_h100] [HOLD_TEACHERS=1] bash submit.sh [pospischil] [hh]
set -euo pipefail
set -x
MODELS="${*:-pospischil hh}"
HOLD=()
if [ -n "${HOLD_TEACHERS:-}" ]; then HOLD=(-H); fi
QUEUE="${QUEUE:-gpu_rtx6000}"
case "$QUEUE" in
  gpu_h100|gpu_rtx6000) SLOTS=12 ;;  # one GPU's share; more slots per GPU delays the submission
  *) echo "QUEUE must be gpu_h100 or gpu_rtx6000" >&2; exit 2 ;;
esac
EXP=/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/complex-neurons
S=$(/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools/snapshot.sh)
C=$S/examples/activitome-connectome/complex-neurons/configs
ENV="set -o pipefail; export OMP_NUM_THREADS=$SLOTS PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d);"
RUN="~/myvenv/bin/python -P -m jaxley_pytorch.training --output-root $EXP/runs --config"
mkdir -p $EXP/runs $EXP/jobs
for M in $MODELS; do
  case "$M" in pospischil|hh) ;; *) echo "unknown model $M" >&2; exit 2 ;; esac
  T=$(bsub -P miaai -q $QUEUE -gpu "num=1" -n $SLOTS -W 30:00 "${HOLD[@]}" -J cn_${M}_n1024_teacher \
    -o $EXP/jobs/${M}_n1024_teacher_%J.log "$ENV $RUN $C/${M}_n1024_teacher.toml" | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
  bsub -P miaai -q $QUEUE -gpu "num=1" -n $SLOTS -W 14:00 -w "done($T)" -J "cn_${M}_n1024_students[1-10]" \
    -o $EXP/jobs/${M}_n1024_students_%J_%I.log "$ENV $RUN $C/${M}_n1024_students.toml --seed \$LSB_JOBINDEX"
  bsub -P miaai -q $QUEUE -gpu "num=1" -n $SLOTS -W 14:00 -w "done($T)" -J cn_${M}_n1024_control \
    -o $EXP/jobs/${M}_n1024_control_%J.log "$ENV $RUN $C/${M}_n1024_control.toml"
done
