#!/usr/bin/env bash

NUM_GPUS=${1:-1}

mpirun -np ${NUM_GPUS} \
    -H localhost:${NUM_GPUS} \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH \
    -mca pml ob1 -mca btl ^openib \
    --allow-run-as-root \
    `python3 train_horovod.py --save_path tmp` |& grep -v "Read -1"