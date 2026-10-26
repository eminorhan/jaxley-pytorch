#!/bin/bash
# Per network size: the teacher (one H100); its ten students (a job array) and the
# control, which start when the teacher is done; the summary (one CPU job) when all
# students have ended. All from a snapshot of the repo; the runs go to
# runs/<experiment_name>/seed<k>. A run that stopped early continues with the same
# command plus --resume. Usage: bash submit.sh
set -euo pipefail
set -x
EXP=/nrs/scicompsoft/orhane/jaxley-pytorch-experiments/activitome-connectome/network-size
S=$(/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools/snapshot.sh)
D=$S/examples/activitome-connectome/network-size
C=$D/configs
mkdir -p $EXP/runs $EXP/jobs $EXP/figures

# Teachers.
T512=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 16:00 -J ns_n512_teacher \
  -o $EXP/jobs/n512_teacher_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --config $C/n512_teacher.toml --output-root $EXP/runs" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
T1024=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 48:00 -J ns_n1024_teacher \
  -o $EXP/jobs/n1024_teacher_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --config $C/n1024_teacher.toml --output-root $EXP/runs" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
T2048=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 168:00 -J ns_n2048_teacher \
  -o $EXP/jobs/n2048_teacher_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --config $C/n2048_teacher.toml --output-root $EXP/runs" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')

# Students: seeds 1-10 per size, from shuffled gains and biases.
S512=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 8:00 -J "ns_n512_students[1-10]" -w "done($T512)" \
  -o $EXP/jobs/n512_students_%J_%I.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --config $C/n512_students.toml --output-root $EXP/runs \
     --seed \$LSB_JOBINDEX" | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
S1024=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 24:00 -J "ns_n1024_students[1-10]" -w "done($T1024)" \
  -o $EXP/jobs/n1024_students_%J_%I.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --config $C/n1024_students.toml --output-root $EXP/runs \
     --seed \$LSB_JOBINDEX" | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
S2048=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 72:00 -J "ns_n2048_students[1-10]" -w "done($T2048)" \
  -o $EXP/jobs/n2048_students_%J_%I.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --config $C/n2048_students.toml --output-root $EXP/runs \
     --seed \$LSB_JOBINDEX" | sed -n 's/.*<\([0-9]*\)>.*/\1/p')

# Controls: from the teacher's own gains and biases.
C512=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 8:00 -J ns_n512_control -w "done($T512)" \
  -o $EXP/jobs/n512_control_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --config $C/n512_control.toml --output-root $EXP/runs" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
C1024=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 24:00 -J ns_n1024_control -w "done($T1024)" \
  -o $EXP/jobs/n1024_control_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --config $C/n1024_control.toml --output-root $EXP/runs" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
C2048=$(bsub -P miaai -q gpu_h100 -gpu "num=1" -n 12 -W 72:00 -J ns_n2048_control -w "done($T2048)" \
  -o $EXP/jobs/n2048_control_%J.log \
  "set -o pipefail; export OMP_NUM_THREADS=12 PYTHONPATH=$S PYTHONPYCACHEPREFIX=\$(mktemp -d); \
   ~/myvenv/bin/python -P -m jaxley_pytorch.training --config $C/n2048_control.toml --output-root $EXP/runs" \
  | sed -n 's/.*<\([0-9]*\)>.*/\1/p')

# Summary: tables and figures.
bsub -P miaai -n 2 -W 0:30 -J ns_summary \
  -w "ended($S512) && ended($S1024) && ended($S2048) && ended($C512) && ended($C1024) && ended($C2048)" \
  -o $EXP/jobs/summary_%J.log \
  "set -o pipefail; export PYTHONPATH=$S; ~/myvenv/bin/python -P $D/summarize.py --runs $EXP/runs \
     --sizes 512 1024 2048 --out $EXP/figures"
