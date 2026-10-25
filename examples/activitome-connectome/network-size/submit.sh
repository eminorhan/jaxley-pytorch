#!/bin/bash
# Per network size: the teacher (one H100); ten students (a job array) and one control
# student, which start when their teacher is done; the summary (one CPU job) when all
# students have ended. All from a snapshot of the repo. Usage: bash submit.sh
set -euo pipefail
set -x
EXP=/nrs/scicompsoft/orhane/probes/jaxley_pytorch/activitome-connectome/network-size
S=$(/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools/snapshot.sh)
TC=$S/examples/activitome-connectome/task-complexity
D=$S/examples/activitome-connectome/network-size
mkdir -p $EXP/runs $EXP/jobs $EXP/figures

# Teachers.
T512=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 16:00 -J ns_teacher_n512 \
  -o $EXP/jobs/teacher_n512_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python $TC/train.py --role teacher --task tokens --n 512 --out $EXP/runs/n512/teacher" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
T1024=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 48:00 -J ns_teacher_n1024 \
  -o $EXP/jobs/teacher_n1024_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python $TC/train.py --role teacher --task tokens --n 1024 --out $EXP/runs/n1024/teacher" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
T2048=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 168:00 -J ns_teacher_n2048 \
  -o $EXP/jobs/teacher_n2048_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python $TC/train.py --role teacher --task tokens --n 2048 --out $EXP/runs/n2048/teacher" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')

# Students: seeds 1-10 per size, from shuffled gains and biases.
S512=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 8:00 -J "ns_students_n512[1-10]" \
  -w "done($T512)" -o $EXP/jobs/student_n512_%J_%I.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python $TC/train.py --role student --task tokens --teacher $EXP/runs/n512/teacher \
     --seed \$LSB_JOBINDEX --out $EXP/runs/n512/student_seed\$LSB_JOBINDEX" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
S1024=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 24:00 -J "ns_students_n1024[1-10]" \
  -w "done($T1024)" -o $EXP/jobs/student_n1024_%J_%I.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python $TC/train.py --role student --task tokens --teacher $EXP/runs/n1024/teacher \
     --seed \$LSB_JOBINDEX --out $EXP/runs/n1024/student_seed\$LSB_JOBINDEX" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
S2048=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 72:00 -J "ns_students_n2048[1-10]" \
  -w "done($T2048)" -o $EXP/jobs/student_n2048_%J_%I.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python $TC/train.py --role student --task tokens --teacher $EXP/runs/n2048/teacher \
     --seed \$LSB_JOBINDEX --out $EXP/runs/n2048/student_seed\$LSB_JOBINDEX" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')

# Control students: one per size, from the teacher's own gains and biases.
C512=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 8:00 -J ns_control_n512 \
  -w "done($T512)" -o $EXP/jobs/control_n512_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python $TC/train.py --role student --task tokens --teacher $EXP/runs/n512/teacher \
     --init teacher --out $EXP/runs/n512/student_control" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
C1024=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 24:00 -J ns_control_n1024 \
  -w "done($T1024)" -o $EXP/jobs/control_n1024_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python $TC/train.py --role student --task tokens --teacher $EXP/runs/n1024/teacher \
     --init teacher --out $EXP/runs/n1024/student_control" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
C2048=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 72:00 -J ns_control_n2048 \
  -w "done($T2048)" -o $EXP/jobs/control_n2048_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python $TC/train.py --role student --task tokens --teacher $EXP/runs/n2048/teacher \
     --init teacher --out $EXP/runs/n2048/student_control" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')

# Summary: tables and figures.
bsub -P miaai -n 2 -W 0:30 -J ns_summary \
  -w "ended($S512) && ended($S1024) && ended($S2048) && ended($C512) && ended($C1024) && ended($C2048)" \
  -o $EXP/jobs/summary_%J.log \
  "set -o pipefail; ~/myvenv/bin/python -I $D/summarize.py \
     --dirs $EXP/runs/n512 $EXP/runs/n1024 $EXP/runs/n2048 --out $EXP/figures"
