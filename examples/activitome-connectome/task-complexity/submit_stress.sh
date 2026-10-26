#!/bin/bash
# The stress test of the cycling result (README.md, "Stress test") with
# jaxley_pytorch.training: five student strategies (configs/cycling_*), ten students each
# (job arrays, one H100 per student). All from a snapshot of the repo; the runs go to
# runs/<experiment_name>/seed<k>, after those of submit.sh (cycling_teacher, and
# cycling_students for cycling_shuffled_lbfgs). Usage: bash submit_stress.sh
set -euo pipefail
set -x
EXP=/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/task-complexity
S=$(/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools/snapshot.sh)
C=$S/examples/activitome-connectome/task-complexity/configs
mkdir -p $EXP/runs $EXP/jobs

# Random starts drawn again while they run away; 15,000 Adam steps, then 500 L-BFGS steps.
bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 2:00 -J "tc_cycling_rstable_lbfgs[1-10]" \
  -o $EXP/jobs/cycling_rstable_lbfgs_%J_%I.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --config $C/cycling_rstable_lbfgs.toml --output-root $EXP/runs \
     --seed \$LSB_JOBINDEX"

# Uniform starts (gains x 0.9 while they run away); then 500 L-BFGS steps.
bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 2:00 -J "tc_cycling_uniform_lbfgs[1-10]" \
  -o $EXP/jobs/cycling_uniform_lbfgs_%J_%I.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --config $C/cycling_uniform_lbfgs.toml --output-root $EXP/runs \
     --seed \$LSB_JOBINDEX"

# Stable random starts trained on 6,400 new trials of the task.
bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 2:00 -J "tc_cycling_rstable_fresh[1-10]" \
  -o $EXP/jobs/cycling_rstable_fresh_%J_%I.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --config $C/cycling_rstable_fresh.toml --output-root $EXP/runs \
     --seed \$LSB_JOBINDEX"

# The same, 30,000 steps at batch 128.
bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 4:00 -J "tc_cycling_rstable_fresh_long[1-10]" \
  -o $EXP/jobs/cycling_rstable_fresh_long_%J_%I.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --config $C/cycling_rstable_fresh_long.toml --output-root $EXP/runs \
     --seed \$LSB_JOBINDEX"

# The shuffled-start students continued with 500 L-BFGS steps (8 of 10 ran into this
# one-hour limit in the original runs; their last evaluation counts).
bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 1:00 -J "tc_cycling_shuffled_lbfgs[1-10]" \
  -o $EXP/jobs/cycling_shuffled_lbfgs_%J_%I.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --config $C/cycling_shuffled_lbfgs.toml --output-root $EXP/runs \
     --seed \$LSB_JOBINDEX"
