#!/bin/bash
# Stress test of the cycling result (README.md, "Stress test"): four student strategies,
# ten students each (job arrays, one H100 per student), then the summary (one CPU job).
# All from a snapshot of the repo. Usage: bash submit_stress.sh
set -euo pipefail
set -x
EXP=/nrs/scicompsoft/orhane/probes/jaxley_pytorch/activitome-connectome/task-complexity
S=$(/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools/snapshot.sh)
D=$S/examples/activitome-connectome/task-complexity

# 1. Random start (the teacher's initial distribution), drawn again while its activity
#    runs away; 15,000 Adam steps, then 500 full-batch L-BFGS steps.
A=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 2:00 -J "tc_stress_rstable[1-10]" \
  -o $EXP/jobs/stress_rstable_%J_%I.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python $D/train.py --role student --task cycling --teacher $EXP/runs/cycling/teacher \
     --init random --stable-start 1 --lbfgs-steps 500 \
     --seed \$LSB_JOBINDEX --out $EXP/runs/cycling/student_rstable_seed\$LSB_JOBINDEX" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')

# 2. Uniform start (gains 1, biases 0; gains scaled by 0.9 while the activity runs away);
#    15,000 Adam steps, then 500 L-BFGS steps.
B=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 2:00 -J "tc_stress_uniform[1-10]" \
  -o $EXP/jobs/stress_uniform_%J_%I.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python $D/train.py --role student --task cycling --teacher $EXP/runs/cycling/teacher \
     --init uniform --stable-start 1 --lbfgs-steps 500 \
     --seed \$LSB_JOBINDEX --out $EXP/runs/cycling/student_uniform_seed\$LSB_JOBINDEX" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')

# 3. Random start, stable draws, trained on 6,400 new trials of the task instead of the 60.
C=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 2:00 -J "tc_stress_rstable_fresh[1-10]" \
  -o $EXP/jobs/stress_rstable_fresh_%J_%I.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python $D/train.py --role student --task cycling --teacher $EXP/runs/cycling/teacher \
     --init random --stable-start 1 --fresh-trials 6400 \
     --seed \$LSB_JOBINDEX --out $EXP/runs/cycling/student_rstable_fresh_seed\$LSB_JOBINDEX" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')

# 4. The shuffled-start students (the paper's), continued with 500 L-BFGS steps.
E=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 1:00 -J "tc_stress_shuffled_lbfgs[1-10]" \
  -o $EXP/jobs/stress_shuffled_lbfgs_%J_%I.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python $D/train.py --role student --task cycling --teacher $EXP/runs/cycling/teacher \
     --init-from $EXP/runs/cycling/student_seed\$LSB_JOBINDEX --steps 0 --lbfgs-steps 500 \
     --seed \$LSB_JOBINDEX --out $EXP/runs/cycling/student_shuffled_lbfgs_seed\$LSB_JOBINDEX" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')

# Summary: stress_test.md and stress_test.png.
bsub -P miaai -n 2 -W 0:15 -J tc_stress_summary -w "ended($A) && ended($B) && ended($C) && ended($E)" \
  -o $EXP/jobs/stress_summary_%J.log \
  "set -o pipefail; ~/myvenv/bin/python -I $D/stress_test.py --runs $EXP/runs --out $EXP/figures"
