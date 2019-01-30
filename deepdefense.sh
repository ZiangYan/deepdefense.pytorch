#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

DIR="output/debug/`date +'%Y-%m-%d_%H:%M:%S'`"
R=$(head -c 500 /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)
DIR=${DIR}"-"${R}
mkdir -p $DIR
LOG=${DIR}"/train.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python3 deepdefense.py --exp-dir $DIR $@
