#!/bin/bash

if [ -z "$3" ]; then
    echo "target 'local' selected automatically."
    target="local"
    tag=$1
    reps=$2
else
   target=$1
   tag=$2
   reps=$3
fi 

seed=`od -A n -t d -N 4 /dev/urandom`
cmd_line=" --algo NAF --env-name HalfCheetah-v2 --seed ${seed}"

${NAF_IKOSTRIKOV_PATH}/exp_scripts_alt/run.sh "${target}" "${cmd_line}" "${tag}" "${reps}"
