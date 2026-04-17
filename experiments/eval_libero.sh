#!/usr/bin/bash

gpu=${1:-1}

config_file=configs/ltx_model/libero/action_model_libero.yaml

output_dir=""

ckpt_path_goal=/PATH/TO/CKPT
ckpt_path_obj=/PATH/TO/CKPT
ckpt_path_10=/PATH/TO/CKPT
ckpt_path_spa=/PATH/TO/CKPT



EGL_DEVICE_ID=$gpu python  experiments/eval_libero.py \
    --config_file $config_file \
    --output_dir  $output_dir \
    --ckpt_path $ckpt_path_10 \
    --exec_step 8 \
    --task_suite_name  libero_10 \
    --device $gpu \
    --num_trails_per_task 50 \
    --threshold 1

