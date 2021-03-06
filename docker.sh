#!/bin/bash
HASH=$1
name=${USER}_naf_ikostrikov_GPU_${GPU}_${HASH}

echo "Launching container named '${name}' on GPU '${GPU}'"
# Launches a docker container using our image, and runs the provided command

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -z "$NAF_IKOSTRIKOV_RESULTS_PATH" ]; then
    RESULTS_PATH="${NAF_IKOSTRIKOV_PATH}/results"
    mkdir -p $RESULTS_PATH
else
    RESULTS_PATH=$NAF_IKOSTRIKOV_RESULTS_PATH
fi

echo "HASH: ${HASH}"
echo "REST: ${@:1}"

echo "RESULTS_PATH: ${RESULTS_PATH}"
${cmd} run -d --rm \
    --name $name \
    --security-opt="apparmor=unconfined" --cap-add=SYS_PTRACE \
    --net host \
    --user root \
    -v $SCRIPT_PATH:/naf_ikostrikov \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v `pwd`/mongodb:/data/db \
    -e DISPLAY=unix$DISPLAY \
    -t naf_ikostrikov \
    ${@:2}

#    -v $RESULTS_PATH:/pymarl/results \

# docker exec -it -u root $name /bin/bash

