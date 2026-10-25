#!/bin/bash
# Smoke test of one network size, inside one GPU job (submit_smoke.sh): 20 steps of a
# teacher, of one student and of the control student, then the equivalence check.
# Usage: bash smoke.sh <neurons> <output dir>
set -euxo pipefail
TC=$(readlink -f "$(dirname "$0")/../task-complexity")
N=$1
OUT=$2/n$N

~/myvenv/bin/python $TC/train.py --role teacher --task tokens --n $N --steps 20 --eval-every 10 --log-every 5 \
  --out $OUT/teacher
~/myvenv/bin/python $TC/train.py --role student --task tokens --teacher $OUT/teacher --seed 1 \
  --steps 20 --eval-every 10 --log-every 5 --out $OUT/student_seed1
~/myvenv/bin/python $TC/train.py --role student --task tokens --teacher $OUT/teacher --init teacher \
  --steps 20 --eval-every 10 --log-every 5 --out $OUT/student_control
~/myvenv/bin/python $TC/train.py --role check --n $N --out $OUT/check
