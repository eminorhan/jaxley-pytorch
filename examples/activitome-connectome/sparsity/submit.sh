#!/bin/bash
# Per network size: the teacher (one H100); its ten students (a job array) and the
# control, which start when the teacher is done; then the summary (one CPU job) of every
# size, with task-complexity's token runs as the 300-neuron size, when the submitted
# students have ended. All from a snapshot of the repo; the runs go to
# runs/<experiment_name>/seed<k>. A run that stopped early continues with the same
# command plus --resume. Time limits: as in ../network-size (the dense simulator's step
# time depends on the number of neurons, not on the connection probability). The summary
# is ../network-size/summarize.py.
# Usage: bash submit.sh [sizes to submit, default: all]
set -euo pipefail
set -x
ALL="600 900"
SIZES="${*:-$ALL}"
declare -A TEACHER_W=([600]=2:00 [900]=3:00) STUDENT_W=([600]=1:00 [900]=1:00)
EXP=/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/sparsity
TC=/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/task-complexity
S=$(/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools/snapshot.sh)
D=$S/examples/activitome-connectome/sparsity
C=$D/configs
ENV="set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d);"
RUN="~/myvenv/bin/python -P -m jaxley_pytorch.training --output-root $EXP/runs --config"
mkdir -p $EXP/runs $EXP/jobs $EXP/figures

ENDED=()
for n in $SIZES; do
  # The teacher; its students (seeds 1-10, from shuffled gains and biases) and the
  # control (from the teacher's own gains and biases) when it is done.
  T=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W ${TEACHER_W[$n]} -J sp_n${n}_teacher \
    -o $EXP/jobs/n${n}_teacher_%J.log "$ENV $RUN $C/n${n}_teacher.toml" | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
  STU=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W ${STUDENT_W[$n]} -J "sp_n${n}_students[1-10]" \
    -w "done($T)" -o $EXP/jobs/n${n}_students_%J_%I.log \
    "$ENV $RUN $C/n${n}_students.toml --seed \$LSB_JOBINDEX" | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
  CON=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W ${STUDENT_W[$n]} -J sp_n${n}_control \
    -w "done($T)" -o $EXP/jobs/n${n}_control_%J.log "$ENV $RUN $C/n${n}_control.toml" \
    | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
  ENDED+=("ended($STU)" "ended($CON)")
done

# Summary: tables and figures of every size.
DEP=$(printf ' && %s' "${ENDED[@]}")
bsub -P miaai -n 2 -W 0:30 -J sp_summary -w "${DEP:4}" -o $EXP/jobs/summary_%J.log \
  "set -o pipefail; export PYTHONPATH=$S; ~/myvenv/bin/python -P $S/examples/activitome-connectome/network-size/summarize.py --runs $EXP/runs \
     --sizes $ALL --task-complexity $TC/runs --out $EXP/figures"
