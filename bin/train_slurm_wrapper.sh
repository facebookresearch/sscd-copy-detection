#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Wraps train.py to set pytorch distributed environment variables with
# values from slurm.

if [ ! -e ./sscd/train.py ]; then
  echo "Run from the top-level sscd directory."
  exit 1
fi

echo "Running on $(hostname)"

# Choose a primary node for distributed coordination
if (( SLURM_STEP_NUM_NODES == 1)); then
  primary="localhost"
else
  primary="$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)"
fi
echo "Using $primary as primary for $SLURM_STEP_NUM_NODES node training run."

MASTER_ADDR="$primary" MASTER_PORT="20285" NODE_RANK="$SLURM_NODEID" WORLD_SIZE=$(( 8 * $SLURM_STEP_NUM_NODES )) \
exec python ./sscd/train.py --nodes="$SLURM_STEP_NUM_NODES" --gpus=8 "$@"
