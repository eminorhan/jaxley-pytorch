#!/bin/bash
# Reruns the students and controls of every experiment with their teacher's learning-rate schedule (2026-11-08). Until
# then every student trained at a peak of 1e-3, for 10,000 steps (rate networks) or 2,000 (spiking networks), and was
# still improving when the rate reached zero (cell-types/probes/student_budget in the output folder). The teachers stay.
# task-complexity's cycling task is not rerun: its teacher trained at 1e-3 too.
# * Rate networks (task-complexity chars and tokens, network-size, sparsity): 25,000 steps at 3e-3. Their earlier
#   student and control runs and figures move to superseded/20261108/ now, and the new runs start now.
# * Spiking networks (complex-neurons, cell-types): 5,000 steps at 3e-3. The move waits until cell-types' 1e-3 round
#   and its summary (which reads complex-neurons' Pospischil runs) have ended, and the new runs wait for the move.
# Then every experiment's summary, as in its submit.sh. One RTX PRO 6000 per run, from a snapshot of the repo.
# Usage: bash rerun_students.sh
set -euo pipefail
set -x
AC=/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome
OLD_CT="155259586 155259587 155259589 155259590 155259592 155259593 155259594"  # cell-types' 1e-3 round + summary
S=$(/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools/snapshot.sh)
E=$S/examples/activitome-connectome
ENV="set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d);"
PY="~/myvenv/bin/python -P"
id() { sed -n 's/.*<\([0-9]*\)>.*/\1/p'; }

note() {  # experiment, old schedule, new schedule, what moved
  mkdir -p $AC/$1/superseded/20261108/runs $AC/$1/superseded/20261108/figures
  cat > $AC/$1/superseded/20261108/NOTE.txt <<EOF
Runs moved out of runs/ on 2026-11-08 (user: "rerun both complex-neurons and cell-types students with the teacher's
schedule, rerun the rate experiments' students with their teachers' schedules as well"): $4, trained at a peak
learning rate of 1e-3 for $2 steps, and the figures summarizing them. They were still improving when the rate reached
zero (cell-types/probes/student_budget). The reruns in runs/ use the teacher's schedule: $3.
EOF
}

# students [1-10] and the control of one group; prints "ended(students) && ended(control)"
group() {  # experiment, name, -W, extra bsub options
  local C=$E/$1/configs RUN="$PY -m jaxley_pytorch.training --output-root $AC/$1/runs --config" STU CON
  STU=$(bsub -P miaai -q gpu_rtx6000 -gpu "num=1" -n 12 -W $3 $4 -J "rerun_$2_students[1-10]" \
    -o $AC/$1/jobs/$2_students_%J_%I.log "$ENV $RUN $C/$2_students.toml --seed \$LSB_JOBINDEX" | id)
  CON=$(bsub -P miaai -q gpu_rtx6000 -gpu "num=1" -n 12 -W $3 $4 -J "rerun_$2_control" \
    -o $AC/$1/jobs/$2_control_%J.log "$ENV $RUN $C/$2_control.toml" | id)
  echo "ended($STU) && ended($CON)"
}

summary() {  # experiment, dependency, command
  bsub -P miaai -n 2 -W 0:30 -J rerun_$1_summary -w "$2" -o $AC/$1/jobs/summary_%J.log \
    "set -o pipefail; export PYTHONPATH=$S; $PY $3"
}

# ---- rate networks: move now, run now
note task-complexity 10,000 "25,000 steps at 3e-3, 1,250 warmup" "the chars and tokens students and controls"
for d in chars_students chars_control tokens_students tokens_control; do
  mv $AC/task-complexity/runs/$d $AC/task-complexity/superseded/20261108/runs/
done
for f in activity_by_task.png outcomes.png summary.json summary.md training_curves.png traces_chars.png traces_tokens.png; do
  mv $AC/task-complexity/figures/$f $AC/task-complexity/superseded/20261108/figures/  # cycling's stay
done
for e in network-size sparsity; do
  note $e 10,000 "25,000 steps at 3e-3, 1,250 warmup" "the students and controls"
  for d in n600_students n600_control n900_students n900_control; do mv $AC/$e/runs/$d $AC/$e/superseded/20261108/runs/; done
  find $AC/$e/figures -maxdepth 1 -type f -exec mv -t $AC/$e/superseded/20261108/figures/ {} +
done
TC_CHARS=$(group task-complexity chars 3:00 "")
TC_TOKENS=$(group task-complexity tokens 3:00 "")
summary task-complexity "$TC_CHARS && $TC_TOKENS" \
  "$E/task-complexity/summarize.py --runs $AC/task-complexity/runs --tasks cycling chars tokens --out $AC/task-complexity/figures"
for e in network-size sparsity; do
  DEP="$TC_TOKENS && $(group $e n600 3:00 "") && $(group $e n900 3:00 "")"  # tokens: the 300-neuron size
  summary $e "$DEP" "$E/network-size/summarize.py --runs $AC/$e/runs --sizes 600 900 \
    --task-complexity $AC/task-complexity/runs --out $AC/$e/figures"
done

# ---- spiking networks: move when cell-types' 1e-3 round and its summary have ended, run after the move
note complex-neurons 2,000 "5,000 steps at 3e-3, 250 warmup" "the stage-2 students and controls"
note cell-types 2,000 "5,000 steps at 3e-3, 250 warmup" \
  "the students and controls (the round was still running when the rerun was decided; it finished first)"
MOVE="set -e; cd $AC/complex-neurons/runs && mv pospischil_n1024_students pospischil_n1024_control hh_n1024_students \
  hh_n1024_control ../superseded/20261108/runs/; find $AC/complex-neurons/figures -maxdepth 1 -type f \
  -exec mv -t $AC/complex-neurons/superseded/20261108/figures/ {} +; cd $AC/cell-types/runs && mv t4_students t4_control \
  t8_students t8_control t16_students t16_control ../superseded/20261108/runs/; find $AC/cell-types/figures -maxdepth 1 \
  -type f -exec mv -t $AC/cell-types/superseded/20261108/figures/ {} +"
WAIT=$(printf ' && ended(%s)' $OLD_CT)
M=$(bsub -P miaai -q short -n 1 -W 0:15 -J rerun_spiking_move -w "${WAIT:4}" -o $AC/cell-types/jobs/rerun_move_%J.log \
  "$MOVE" | id)
CN_P=$(group complex-neurons pospischil_n1024 30:00 "-w done($M)")
CN_H=$(group complex-neurons hh_n1024 30:00 "-w done($M)")
summary complex-neurons "$CN_P && $CN_H" \
  "$E/complex-neurons/summarize.py --runs $AC/complex-neurons/runs --out $AC/complex-neurons/figures"
DEP="$CN_P"
GROUP_ARGS=()
for K in 4 8 16; do
  DEP="$DEP && $(group cell-types t$K 30:00 "-w done($M)")"
  GROUP_ARGS+=(--group "\"$K types\"" $AC/cell-types/runs t$K)
done
summary cell-types "$DEP" "$E/complex-neurons/summarize.py ${GROUP_ARGS[*]} \
  --group \"1,024 types (complex-neurons)\" $AC/complex-neurons/runs pospischil_n1024 --out $AC/cell-types/figures"
