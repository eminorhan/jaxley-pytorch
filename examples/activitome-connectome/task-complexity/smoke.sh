#!/bin/bash
# Smoke test, inside one GPU job (submit_smoke.sh): the equivalence check, a few steps of
# every teacher and of one student per task (and of one control student), then the
# summary of these runs.
# Usage: bash smoke.sh <output dir>
set -euxo pipefail
D=$(dirname "$(readlink -f "$0")")
OUT=$1

~/myvenv/bin/python $D/train.py --role check --out $OUT/check

~/myvenv/bin/python $D/train.py --role teacher --task cycling --steps 20 --eval-every 10 --log-every 5 \
  --out $OUT/cycling/teacher
~/myvenv/bin/python $D/train.py --role student --task cycling --teacher $OUT/cycling/teacher --seed 1 \
  --steps 20 --eval-every 10 --log-every 5 --out $OUT/cycling/student_seed1

~/myvenv/bin/python $D/train.py --role teacher --task chars --steps 40 --eval-every 20 --log-every 5 \
  --out $OUT/chars/teacher
~/myvenv/bin/python $D/train.py --role student --task chars --teacher $OUT/chars/teacher --seed 1 \
  --steps 40 --eval-every 20 --log-every 5 --out $OUT/chars/student_seed1
~/myvenv/bin/python $D/train.py --role student --task chars --teacher $OUT/chars/teacher --init teacher \
  --steps 40 --eval-every 20 --log-every 5 --out $OUT/chars/student_control

~/myvenv/bin/python $D/train.py --role teacher --task tokens --steps 40 --eval-every 20 --log-every 5 \
  --out $OUT/tokens/teacher
~/myvenv/bin/python $D/train.py --role student --task tokens --teacher $OUT/tokens/teacher --seed 1 \
  --steps 40 --eval-every 20 --log-every 5 --out $OUT/tokens/student_seed1

~/myvenv/bin/python $D/summarize.py --runs $OUT --out $OUT/summary
