#!/bin/bash
# Stages 1-2: the three teachers (one H100 each); per task, ten students (a job array) and
# one control student, which start when their teacher is done; the summary (one CPU job)
# when all students have ended. All from a snapshot of the repo. Usage: bash submit.sh
set -euo pipefail
set -x
EXP=/nrs/scicompsoft/orhane/probes/jaxley_pytorch/activitome-connectome/task-complexity
S=$(/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools/snapshot.sh)
D=$S/examples/activitome-connectome/task-complexity
mkdir -p $EXP/runs $EXP/jobs $EXP/figures

# Teachers.
T_CYCLING=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 1:00 -J tc_teacher_cycling \
  -o $EXP/jobs/teacher_cycling_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python $D/train.py --role teacher --task cycling --out $EXP/runs/cycling/teacher" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
T_CHARS=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 8:00 -J tc_teacher_chars \
  -o $EXP/jobs/teacher_chars_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python $D/train.py --role teacher --task chars --out $EXP/runs/chars/teacher" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
T_TOKENS=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 8:00 -J tc_teacher_tokens \
  -o $EXP/jobs/teacher_tokens_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python $D/train.py --role teacher --task tokens --out $EXP/runs/tokens/teacher" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')

# Students: seeds 1-10 per task, from shuffled gains and biases.
S_CYCLING=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 2:00 -J "tc_students_cycling[1-10]" \
  -w "done($T_CYCLING)" -o $EXP/jobs/student_cycling_%J_%I.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python $D/train.py --role student --task cycling --teacher $EXP/runs/cycling/teacher \
     --seed \$LSB_JOBINDEX --out $EXP/runs/cycling/student_seed\$LSB_JOBINDEX" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
S_CHARS=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 4:00 -J "tc_students_chars[1-10]" \
  -w "done($T_CHARS)" -o $EXP/jobs/student_chars_%J_%I.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python $D/train.py --role student --task chars --teacher $EXP/runs/chars/teacher \
     --seed \$LSB_JOBINDEX --out $EXP/runs/chars/student_seed\$LSB_JOBINDEX" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
S_TOKENS=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 4:00 -J "tc_students_tokens[1-10]" \
  -w "done($T_TOKENS)" -o $EXP/jobs/student_tokens_%J_%I.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python $D/train.py --role student --task tokens --teacher $EXP/runs/tokens/teacher \
     --seed \$LSB_JOBINDEX --out $EXP/runs/tokens/student_seed\$LSB_JOBINDEX" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')

# Control students: one per task, from the teacher's own gains and biases.
C_CYCLING=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 2:00 -J tc_control_cycling \
  -w "done($T_CYCLING)" -o $EXP/jobs/control_cycling_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python $D/train.py --role student --task cycling --teacher $EXP/runs/cycling/teacher \
     --init teacher --out $EXP/runs/cycling/student_control" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
C_CHARS=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 4:00 -J tc_control_chars \
  -w "done($T_CHARS)" -o $EXP/jobs/control_chars_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python $D/train.py --role student --task chars --teacher $EXP/runs/chars/teacher \
     --init teacher --out $EXP/runs/chars/student_control" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
C_TOKENS=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 4:00 -J tc_control_tokens \
  -w "done($T_TOKENS)" -o $EXP/jobs/control_tokens_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python $D/train.py --role student --task tokens --teacher $EXP/runs/tokens/teacher \
     --init teacher --out $EXP/runs/tokens/student_control" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')

# Summary: tables and figure.
bsub -P miaai -n 2 -W 0:30 -J tc_summary \
  -w "ended($S_CYCLING) && ended($S_CHARS) && ended($S_TOKENS) && ended($C_CYCLING) && ended($C_CHARS) && ended($C_TOKENS)" \
  -o $EXP/jobs/summary_%J.log \
  "set -o pipefail; ~/myvenv/bin/python -I $D/summarize.py --runs $EXP/runs --out $EXP/figures"
