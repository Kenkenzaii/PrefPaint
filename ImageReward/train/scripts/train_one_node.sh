#! /bin/bash

NUM_GPUS_PER_WORKER=4
MASTER_PORT=29559

train_options=" \
       --savepath Regre_256_0.01_1w \
       --batch_size 5 \
       --accumulation_steps 1 \
       --epochs 100 \
       --distributed True \
       --gpu_num ${NUM_GPUS_PER_WORKER} \
       --gpu_id '0,1,2,3,4' \
       --clear_visualizer \
       --fix_rate 0.7 \
       --lr 1e-05 \
       --lr-decay-style cosine \
       --warmup 0.0 \
       --rank_pair True\
       --load_pair_store \
       --std_log \
       --valid_per_epoch 4 \
"

run_cmd="torchrun
        --nnodes=1
        --nproc_per_node=${NUM_GPUS_PER_WORKER}
        --master_port=${MASTER_PORT}
        ./src/train_group_regression.py ${train_options}"

echo ${run_cmd}
eval ${run_cmd}
