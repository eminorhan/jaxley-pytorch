#!/bin/bash
# Reruns the text and the spiking experiments at a peak learning rate of 5e-3 instead of 3e-3: teachers, students and
# controls (2026-11-14). Nothing else changes (steps, warmup, seeds), so students and controls keep their teacher's
# schedule. task-complexity's cycling task is not part of it.
# 1. The current runs (teachers, students, controls) and the figures summarizing them move to
#    <experiment>/superseded/20261114/; in complex-neurons and cell-types also the cancelled 5e-3 pilot's students and
#    controls (runs/*_lr5e-3), which started from the 3e-3 teachers.
# 2. Per group: the teacher; when it is done, its ten students (a job array) and the control.
# 3. Every experiment's summary when its students and controls have ended (network-size and sparsity also wait for
#    task-complexity's token runs, their 300-neuron size; cell-types for complex-neurons' Pospischil runs, its
#    per-neuron reference), and the trace figures.
# One RTX PRO 6000 per run, from a snapshot of the repo.
# Usage: bash rerun_all_lr5e-3.sh
set -euo pipefail
set -x
AC=/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome
D=superseded/20261114
RATE_RUNS="n600_teacher n600_students n600_control n900_teacher n900_students n900_control"
TC_RUNS="chars_teacher chars_students chars_control tokens_teacher tokens_students tokens_control"
TC_FIGS="activity_by_task.png outcomes.png summary.json summary.md training_curves.png traces_chars.png traces_tokens.png"
CN_RUNS="pospischil_n1024_teacher pospischil_n1024_students pospischil_n1024_control hh_n1024_teacher hh_n1024_students
  hh_n1024_control pospischil_n1024_students_lr5e-3 pospischil_n1024_control_lr5e-3"
CT_RUNS=$(for k in 4 8 16; do echo t${k}_teacher t${k}_students t${k}_control t${k}_students_lr5e-3 t${k}_control_lr5e-3; done)

# ---- nothing is moved unless everything to move is there and nothing was moved before
for e in task-complexity network-size sparsity complex-neurons cell-types; do
  [ ! -e $AC/$e/$D ] || { echo "$AC/$e/$D exists" >&2; exit 1; }
done
check() { local e=$1; shift; for d in "$@"; do [ -d $AC/$e/runs/$d ] || { echo "missing $AC/$e/runs/$d" >&2; exit 1; }; done; }
check task-complexity $TC_RUNS; check network-size $RATE_RUNS; check sparsity $RATE_RUNS
check complex-neurons $CN_RUNS; check cell-types $CT_RUNS
for f in $TC_FIGS; do [ -f $AC/task-complexity/figures/$f ] || { echo "missing figure $f" >&2; exit 1; }; done

S=$(/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools/snapshot.sh)
E=$S/examples/activitome-connectome
ENV="set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d);"
PY="~/myvenv/bin/python -P"
id() { sed -n 's/.*<\([0-9]*\)>.*/\1/p'; }

# ---- 1. the 3e-3 runs and their figures to superseded/20261114
archive() {  # experiment, "runs", a sentence more for the note; the figures are moved by the caller
  mkdir -p $AC/$1/$D/runs $AC/$1/$D/figures
  for d in $2; do mv $AC/$1/runs/$d $AC/$1/$D/runs/; done
  cat > $AC/$1/$D/NOTE.txt <<EOF
Runs moved out of runs/ on 2026-11-14 (user: "re-run ... with peak learning 5e-3 instead of 3e-3 (both teachers and
students/controls)? No other changes in the configs"): the teachers, students and controls trained at a peak learning
rate of 3e-3 (students and controls with their teacher's schedule, since 2026-11-08), and the figures summarizing them.
$3
The runs in runs/ use 5e-3 with the same steps, warmup and seeds (rerun_all_lr5e-3.sh in the repo).
EOF
}
archive task-complexity "$TC_RUNS" "Only the chars and tokens runs moved; the cycling runs stay."
for f in $TC_FIGS; do mv $AC/task-complexity/figures/$f $AC/task-complexity/$D/figures/; done
for e in network-size sparsity; do
  archive $e "$RATE_RUNS" ""
  find $AC/$e/figures -maxdepth 1 -type f -exec mv -t $AC/$e/$D/figures/ {} +
done
archive complex-neurons "$CN_RUNS" "Also the cancelled 5e-3 pilot's student and control (pospischil_n1024_*_lr5e-3:
2,000 steps at 5e-3 from the 3e-3 teacher)."
find $AC/complex-neurons/figures -maxdepth 1 -type f -exec mv -t $AC/complex-neurons/$D/figures/ {} +
archive cell-types "$CT_RUNS" "Also the cancelled 5e-3 pilot's students and controls (t*_lr5e-3: 2,000 steps at 5e-3
from the 3e-3 teachers) and their summary (figures/lr5e-3)."
find $AC/cell-types/figures -maxdepth 1 -mindepth 1 -exec mv -t $AC/cell-types/$D/figures/ {} +

# ---- 2. per group: the teacher, then its students [1-10] and the control; prints "ended(students) && ended(control)"
group() {  # experiment, tag, name, the teacher's -W, the students' and the control's -W
  local C=$E/$1/configs RUN="$PY -m jaxley_pytorch.training --output-root $AC/$1/runs --config" T STU CON
  local B="bsub -P miaai -q gpu_rtx6000 -gpu num=1 -n 12"
  T=$($B -W $4 -J "lr5e-3_$2_$3_teacher" -o $AC/$1/jobs/$3_teacher_%J.log "$ENV $RUN $C/$3_teacher.toml" | id)
  STU=$($B -W $5 -w "done($T)" -J "lr5e-3_$2_$3_students[1-10]" -o $AC/$1/jobs/$3_students_%J_%I.log \
    "$ENV $RUN $C/$3_students.toml --seed \$LSB_JOBINDEX" | id)
  CON=$($B -W $5 -w "done($T)" -J "lr5e-3_$2_$3_control" -o $AC/$1/jobs/$3_control_%J.log \
    "$ENV $RUN $C/$3_control.toml" | id)
  echo "ended($STU) && ended($CON)"
}
summary() {  # experiment, dependency, command
  bsub -P miaai -n 2 -W 0:30 -J lr5e-3_$1_summary -w "$2" -o $AC/$1/jobs/summary_%J.log \
    "set -o pipefail; export PYTHONPATH=$S; $PY $3"
}

TC_CHARS=$(group task-complexity tc chars 8:00 4:00)
TC_TOKENS=$(group task-complexity tc tokens 8:00 4:00)
NS_600=$(group network-size ns n600 2:00 2:00)
NS_900=$(group network-size ns n900 3:00 3:00)
SP_600=$(group sparsity sp n600 2:00 2:00)
SP_900=$(group sparsity sp n900 3:00 3:00)
CN_P=$(group complex-neurons cn pospischil_n1024 30:00 30:00)
CN_H=$(group complex-neurons cn hh_n1024 30:00 30:00)
CT_4=$(group cell-types ct t4 30:00 30:00)
CT_8=$(group cell-types ct t8 30:00 30:00)
CT_16=$(group cell-types ct t16 30:00 30:00)

# ---- 3. summaries and trace figures
summary task-complexity "$TC_CHARS && $TC_TOKENS" \
  "$E/task-complexity/summarize.py --runs $AC/task-complexity/runs --tasks cycling chars tokens --out $AC/task-complexity/figures"
summary network-size "$TC_TOKENS && $NS_600 && $NS_900" "$E/network-size/summarize.py --runs $AC/network-size/runs \
  --sizes 600 900 --task-complexity $AC/task-complexity/runs --out $AC/network-size/figures"
summary sparsity "$TC_TOKENS && $SP_600 && $SP_900" "$E/network-size/summarize.py --runs $AC/sparsity/runs \
  --sizes 600 900 --task-complexity $AC/task-complexity/runs --out $AC/sparsity/figures"
summary complex-neurons "$CN_P && $CN_H" \
  "$E/complex-neurons/summarize.py --runs $AC/complex-neurons/runs --out $AC/complex-neurons/figures"
summary cell-types "$CN_P && $CT_4 && $CT_8 && $CT_16" "$E/complex-neurons/summarize.py \
  --group \"4 types\" $AC/cell-types/runs t4 --group \"8 types\" $AC/cell-types/runs t8 \
  --group \"16 types\" $AC/cell-types/runs t16 \
  --group \"1,024 types (complex-neurons)\" $AC/complex-neurons/runs pospischil_n1024 \
  --out $AC/cell-types/figures --title \"cell-types (1,024 neurons)\""

T="$PY $E/task-complexity/traces.py"
bsub -P miaai -q short -n 8 -W 0:30 -J lr5e-3_traces_rate \
  -w "$TC_CHARS && $TC_TOKENS && $NS_600 && $NS_900 && $SP_600 && $SP_900" -o $AC/task-complexity/jobs/traces_%J.log \
  "set -x; set -e; export PYTHONPATH=$S OMP_NUM_THREADS=8; \
   $T --runs $AC/task-complexity/runs --out $AC/task-complexity/figures --task chars; \
   $T --runs $AC/task-complexity/runs --out $AC/task-complexity/figures --task tokens; \
   $T --runs $AC/network-size/runs --out $AC/network-size/figures --task tokens --name n600 --configs $E/network-size/configs; \
   $T --runs $AC/network-size/runs --out $AC/network-size/figures --task tokens --name n900 --configs $E/network-size/configs; \
   $T --runs $AC/sparsity/runs --out $AC/sparsity/figures --task tokens --name n600 --configs $E/sparsity/configs; \
   $T --runs $AC/sparsity/runs --out $AC/sparsity/figures --task tokens --name n900 --configs $E/sparsity/configs"
bsub -P miaai -q gpu_rtx6000 -gpu "num=1" -n 12 -W 0:30 -J lr5e-3_traces_spiking -w "$CN_P && $CN_H" \
  -o $AC/complex-neurons/jobs/traces_%J.log \
  "set -x; set -e; export PYTHONPATH=$S OMP_NUM_THREADS=12 PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   $T --runs $AC/complex-neurons/runs --out $AC/complex-neurons/figures --task tokens --name pospischil_n1024 --configs $E/complex-neurons/configs --device cuda; \
   $T --runs $AC/complex-neurons/runs --out $AC/complex-neurons/figures --task tokens --name hh_n1024 --configs $E/complex-neurons/configs --device cuda"
