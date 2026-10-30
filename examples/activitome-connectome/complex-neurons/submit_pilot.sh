#!/bin/bash
# Stage 1, the pilot (1,024 neurons, one GPU each: an H100, or with QUEUE=gpu_rtx6000 an RTX PRO 6000), per model
# (pospischil or hh), configs in configs/pilots. `teachers`: the teacher at peak learning rates 3e-4, 1e-3 and 3e-3
# (pilot_<model>_lr*_teacher.toml), 2,000 steps each. `students`: three students (seeds 1-3) and the control of the
# teacher that pilot_<model>_students.toml and pilot_<model>_control.toml name (the one with the lowest validation loss
# at the end), 1,000 steps each. `students_lr` and `random_lr`: a student from shuffled (pilot_<model>_students_lr*.toml)
# or random (pilot_<model>_random_lr*.toml) single-neuron parameters at peak learning rates 3e-3 and 1e-2, 2,000 steps
# each: where does its activity go once it closes the task-loss gap? `teachers_batch`: the lr-3e-3 teacher with 64 and
# 128 streams per batch (pilot_<model>_lr3e-3_teacher_b*.toml), 2,000 steps each: is the batch size the bottleneck?
# `variant <name>...`: the lr-3e-3 teacher with a change, pilot_<model>_lr3e-3_teacher_<name>.toml (window128: the
# gradient cut every 128 tokens; release5mV: synaptic release over 5 mV; surrogate0.1: the surrogate gradient of
# na_activation_gradient = 0.1), batch 32, 2,000 steps (12 h; 14 s per step on an RTX PRO 6000). With AFTER=<job id>, the jobs start when that job is done. From a snapshot of the repo; the runs go to
# runs/<experiment_name>/seed<k>. A run that stopped early continues with the same command plus --resume. Time limits:
# teachers about 1.4x the H100 smokes' step times (Pospischil 22.4 s, Hodgkin-Huxley 20 s), at batch 64 and 128 about
# 1.4x the RTX PRO 6000's (25.2 s and 45.5 s, Pospischil); students 14 h (on an RTX PRO 6000, 1,000 steps took 4.4 h
# for Pospischil and 3.8 h for Hodgkin-Huxley, 2,000 about 8.6 and 7.4 h).
# Usage: [AFTER=<job id>] [QUEUE=gpu_rtx6000] bash submit_pilot.sh pospischil|hh \
#   teachers|students|students_lr|random_lr|teachers_batch|variant <name>...
set -euo pipefail
set -x
USAGE="usage: bash submit_pilot.sh pospischil|hh teachers|students|students_lr|random_lr|teachers_batch|variant <name>..."
MODEL="${1:-}"
case "$MODEL" in pospischil|hh) ;; *) echo "$USAGE" >&2; exit 2 ;; esac
QUEUE="${QUEUE:-gpu_h100}"
case "$QUEUE" in
  gpu_h100|gpu_rtx6000) SLOTS=12 ;;  # one GPU's share; more slots per GPU delays the submission
  *) echo "QUEUE must be gpu_h100 or gpu_rtx6000" >&2; exit 2 ;;
esac
EXP=/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/complex-neurons
S=$(/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools/snapshot.sh)
C=$S/examples/activitome-connectome/complex-neurons/configs/pilots
ENV="set -o pipefail; export OMP_NUM_THREADS=$SLOTS PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d);"
RUN="~/myvenv/bin/python -P -m jaxley_pytorch.training --output-root $EXP/runs --config"
DEP=()
if [ -n "${AFTER:-}" ]; then DEP=(-w "done($AFTER)"); fi
mkdir -p $EXP/runs $EXP/jobs

case "${2:-}" in
  teachers)
    for lr in 3e-4 1e-3 3e-3; do
      bsub -P miaai -q $QUEUE -gpu "num=1" -n $SLOTS -W 18:00 "${DEP[@]}" -J cn_pilot_${MODEL}_lr${lr}_teacher \
        -o $EXP/jobs/pilot_${MODEL}_lr${lr}_teacher_%J.log "$ENV $RUN $C/pilot_${MODEL}_lr${lr}_teacher.toml"
    done ;;
  students)
    bsub -P miaai -q $QUEUE -gpu "num=1" -n $SLOTS -W 14:00 "${DEP[@]}" -J "cn_pilot_${MODEL}_students[1-3]" \
      -o $EXP/jobs/pilot_${MODEL}_students_%J_%I.log \
      "$ENV $RUN $C/pilot_${MODEL}_students.toml --seed \$LSB_JOBINDEX"
    bsub -P miaai -q $QUEUE -gpu "num=1" -n $SLOTS -W 14:00 "${DEP[@]}" -J cn_pilot_${MODEL}_control \
      -o $EXP/jobs/pilot_${MODEL}_control_%J.log "$ENV $RUN $C/pilot_${MODEL}_control.toml" ;;
  teachers_batch)
    declare -A W=([64]=20:00 [128]=36:00)
    for B in 64 128; do
      CFG=$C/pilot_${MODEL}_lr3e-3_teacher_b$B.toml
      [ -f $CFG ] || { echo "no config $CFG" >&2; exit 2; }
      bsub -P miaai -q $QUEUE -gpu "num=1" -n $SLOTS -W ${W[$B]} "${DEP[@]}" -J cn_pilot_${MODEL}_lr3e-3_teacher_b$B \
        -o $EXP/jobs/pilot_${MODEL}_lr3e-3_teacher_b${B}_%J.log "$ENV $RUN $CFG"
    done ;;
  variant)
    for V in "${@:3}"; do
      CFG=$C/pilot_${MODEL}_lr3e-3_teacher_$V.toml
      [ -f $CFG ] || { echo "no config $CFG" >&2; exit 2; }
      bsub -P miaai -q $QUEUE -gpu "num=1" -n $SLOTS -W 12:00 "${DEP[@]}" -J cn_pilot_${MODEL}_lr3e-3_teacher_$V \
        -o $EXP/jobs/pilot_${MODEL}_lr3e-3_teacher_${V}_%J.log "$ENV $RUN $CFG"
    done ;;
  students_lr|random_lr)
    KIND=$([ "$2" = students_lr ] && echo students || echo random)
    for lr in 3e-3 1e-2; do
      bsub -P miaai -q $QUEUE -gpu "num=1" -n $SLOTS -W 14:00 "${DEP[@]}" -J cn_pilot_${MODEL}_${KIND}_lr${lr} \
        -o $EXP/jobs/pilot_${MODEL}_${KIND}_lr${lr}_%J.log "$ENV $RUN $C/pilot_${MODEL}_${KIND}_lr${lr}.toml --seed 1"
    done ;;
  *)
    echo "$USAGE" >&2
    exit 2 ;;
esac
