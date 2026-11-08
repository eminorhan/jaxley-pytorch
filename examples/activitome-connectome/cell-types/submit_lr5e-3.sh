#!/bin/bash
# The second round of students at a peak learning rate of 5e-3 (configs/t<K>_{students,control}_lr5e-3.toml and the
# per-neuron reference ../complex-neurons/configs/pospischil_n1024_{students,control}_lr5e-3.toml): the same starts and
# seeds as the first round (1e-3), whose students could move every value by only about 1 unit in their 2,000 steps
# (README). The teachers are trained; ten students (job array) and a control per teacher, 2,000 steps each, one RTX
# PRO 6000 each; then the summary of the four groups (../complex-neurons/summarize.py --suffix _lr5e-3) to
# figures/lr5e-3/. From a snapshot of the repo.
# Usage: bash submit_lr5e-3.sh
set -euo pipefail
set -x
EXP=/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/cell-types
CN=/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/complex-neurons
S=$(/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools/snapshot.sh)
ENV="set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d);"
PY="~/myvenv/bin/python -P -m jaxley_pytorch.training"
mkdir -p $EXP/jobs $EXP/figures/lr5e-3 $CN/jobs
ENDED=()
GROUP_ARGS=()
# (name, configs, output root, jobs dir, label) per group
for spec in "t4|$S/examples/activitome-connectome/cell-types/configs|$EXP|4 types" \
            "t8|$S/examples/activitome-connectome/cell-types/configs|$EXP|8 types" \
            "t16|$S/examples/activitome-connectome/cell-types/configs|$EXP|16 types" \
            "pospischil_n1024|$S/examples/activitome-connectome/complex-neurons/configs|$CN|1,024 types (per neuron)"; do
  IFS='|' read -r NAME C OUT LABEL <<< "$spec"
  RUN="$PY --output-root $OUT/runs --config"
  STU=$(bsub -P miaai -q gpu_rtx6000 -gpu "num=1" -n 12 -W 14:00 -J "lr5e-3_${NAME}_students[1-10]" \
    -o $OUT/jobs/${NAME}_students_lr5e-3_%J_%I.log "$ENV $RUN $C/${NAME}_students_lr5e-3.toml --seed \$LSB_JOBINDEX" \
    | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
  CON=$(bsub -P miaai -q gpu_rtx6000 -gpu "num=1" -n 12 -W 14:00 -J "lr5e-3_${NAME}_control" \
    -o $OUT/jobs/${NAME}_control_lr5e-3_%J.log "$ENV $RUN $C/${NAME}_control_lr5e-3.toml" | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
  ENDED+=("ended($STU)" "ended($CON)")
  GROUP_ARGS+=(--group "\"$LABEL\"" $OUT/runs $NAME)
done

DEP=$(printf ' && %s' "${ENDED[@]}")
bsub -P miaai -n 2 -W 0:30 -J lr5e-3_summary -w "${DEP:4}" -o $EXP/jobs/summary_lr5e-3_%J.log \
  "set -o pipefail; export PYTHONPATH=$S; ~/myvenv/bin/python -P $S/examples/activitome-connectome/complex-neurons/summarize.py \
     ${GROUP_ARGS[*]} --suffix _lr5e-3 --out $EXP/figures/lr5e-3"
