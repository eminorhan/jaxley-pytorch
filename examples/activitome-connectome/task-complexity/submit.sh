#!/bin/bash
# The experiment's runs with jaxley_pytorch.training (configs/; the stress test is in
# submit_stress.sh): per task the teacher (one H100); its ten students (a job array) and
# the control, which start when the teacher is done; for the cycling task also the
# random-start students. All from a snapshot of the repo; the runs go to
# runs/<experiment_name>/seed<k>. Usage: bash submit.sh
set -euo pipefail
set -x
EXP=/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/task-complexity
S=$(/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools/snapshot.sh)
C=$S/examples/activitome-connectome/task-complexity/configs
mkdir -p $EXP/runs $EXP/jobs

# Teachers.
T_CYCLING=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 1:00 -J tc_cycling_teacher \
  -o $EXP/jobs/cycling_teacher_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --config $C/cycling_teacher.toml --output-root $EXP/runs" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
T_CHARS=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 8:00 -J tc_chars_teacher \
  -o $EXP/jobs/chars_teacher_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --config $C/chars_teacher.toml --output-root $EXP/runs" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
T_TOKENS=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 8:00 -J tc_tokens_teacher \
  -o $EXP/jobs/tokens_teacher_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --config $C/tokens_teacher.toml --output-root $EXP/runs" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')

# Students: seeds 1-10 per arm.
bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 2:00 -J "tc_cycling_students[1-10]" -w "done($T_CYCLING)" \
  -o $EXP/jobs/cycling_students_%J_%I.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --config $C/cycling_students.toml --output-root $EXP/runs \
     --seed \$LSB_JOBINDEX"
bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 2:00 -J "tc_cycling_students_random[1-10]" -w "done($T_CYCLING)" \
  -o $EXP/jobs/cycling_students_random_%J_%I.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --config $C/cycling_students_random.toml --output-root $EXP/runs \
     --seed \$LSB_JOBINDEX"
bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 4:00 -J "tc_chars_students[1-10]" -w "done($T_CHARS)" \
  -o $EXP/jobs/chars_students_%J_%I.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --config $C/chars_students.toml --output-root $EXP/runs \
     --seed \$LSB_JOBINDEX"
bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 4:00 -J "tc_tokens_students[1-10]" -w "done($T_TOKENS)" \
  -o $EXP/jobs/tokens_students_%J_%I.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --config $C/tokens_students.toml --output-root $EXP/runs \
     --seed \$LSB_JOBINDEX"

# Controls: from the teacher's own gains and biases.
bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 2:00 -J tc_cycling_control -w "done($T_CYCLING)" \
  -o $EXP/jobs/cycling_control_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --config $C/cycling_control.toml --output-root $EXP/runs"
bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 4:00 -J tc_chars_control -w "done($T_CHARS)" \
  -o $EXP/jobs/chars_control_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --config $C/chars_control.toml --output-root $EXP/runs"
bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 4:00 -J tc_tokens_control -w "done($T_TOKENS)" \
  -o $EXP/jobs/tokens_control_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --config $C/tokens_control.toml --output-root $EXP/runs"
