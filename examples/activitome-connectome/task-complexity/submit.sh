#!/bin/bash
# The experiment's runs with jaxley_pytorch.training (configs/; the stress test is in submit_stress.sh): per task named
# (cycling, chars, tokens) the teacher; when it is done, its ten students (a job array) and the control; for the
# cycling task also the random-start students. One GPU each: an RTX PRO 6000 (QUEUE=gpu_h100 for an H100). All from a
# snapshot of the repo; the runs go to runs/<experiment_name>/seed<k>; a run that stopped early continues with the same
# command plus --resume.
# Usage: [QUEUE=gpu_h100] bash submit.sh <task>...   (e.g. bash submit.sh chars tokens)
set -euo pipefail
set -x
[ $# -gt 0 ] || { echo "usage: bash submit.sh <task>... (cycling, chars, tokens)" >&2; exit 2; }
QUEUE="${QUEUE:-gpu_rtx6000}"
case "$QUEUE" in gpu_h100|gpu_rtx6000) ;; *) echo "QUEUE must be gpu_h100 or gpu_rtx6000" >&2; exit 2 ;; esac
EXP=/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/task-complexity
S=$(/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools/snapshot.sh)
C=$S/examples/activitome-connectome/task-complexity/configs
ENV="set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d);"
RUN="~/myvenv/bin/python -P -m jaxley_pytorch.training --output-root $EXP/runs --config"
declare -A TEACHER_W=([cycling]=1:00 [chars]=8:00 [tokens]=8:00) STUDENT_W=([cycling]=2:00 [chars]=4:00 [tokens]=4:00)
mkdir -p $EXP/runs $EXP/jobs
for t in "$@"; do
  case "$t" in cycling|chars|tokens) ;; *) echo "unknown task $t" >&2; exit 2 ;; esac
  T=$(bsub -P miaai -q $QUEUE -gpu "num=1" -n 12 -W ${TEACHER_W[$t]} -J tc_${t}_teacher \
    -o $EXP/jobs/${t}_teacher_%J.log "$ENV $RUN $C/${t}_teacher.toml" | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
  ARMS="students"
  if [ $t = cycling ]; then ARMS="students students_random"; fi
  for arm in $ARMS; do
    bsub -P miaai -q $QUEUE -gpu "num=1" -n 12 -W ${STUDENT_W[$t]} -w "done($T)" -J "tc_${t}_${arm}[1-10]" \
      -o $EXP/jobs/${t}_${arm}_%J_%I.log "$ENV $RUN $C/${t}_${arm}.toml --seed \$LSB_JOBINDEX"
  done
  bsub -P miaai -q $QUEUE -gpu "num=1" -n 12 -W ${STUDENT_W[$t]} -w "done($T)" -J tc_${t}_control \
    -o $EXP/jobs/${t}_control_%J.log "$ENV $RUN $C/${t}_control.toml"
done
