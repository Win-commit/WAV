#!/usr/bin/bash

gpu=${1:-1}

config_file=configs/ltx_model/libero/action_model_WV_libero.yaml

output_dir=evaluation_results/libero/Memory_analy/40k/explore_7

ckpt_path_goal=/defaultShare/Genie-Envisioner/lrz_training_v2/libero/Phase3_PolicyModel_WV_absolute_eef/2026_02_14_19_00_36/step_40000/diffusion_pytorch_model.safetensors
ckpt_path_obj=/defaultShare/Genie-Envisioner/lrz_training_v2/libero/Phase3_PolicyModel_WV_absolute_eef/2026_02_14_19_00_36/step_40000/diffusion_pytorch_model.safetensors
ckpt_path_10=/defaultShare/Genie-Envisioner/lrz_training_v2/libero/Phase3_PolicyModel_WV_absolute_eef/2026_02_14_19_00_36/step_40000/diffusion_pytorch_model.safetensors
ckpt_path_spa=/defaultShare/Genie-Envisioner/lrz_training_v2/libero/Phase3_PolicyModel_WV_absolute_eef/2026_02_14_19_00_36/step_40000/diffusion_pytorch_model.safetensors


# EGL_DEVICE_ID=$gpu python  experiments/eval_libero.py \
#     --config_file $config_file \
#     --output_dir  $output_dir \
#     --ckpt_path $ckpt_path_goal \
#     --exec_step 8 \
#     --task_suite_name  libero_goal \
#     --device $gpu \
#     --num_trails_per_task 50 \
#     --threshold 20


EGL_DEVICE_ID=$gpu python  experiments/eval_libero.py \
    --config_file $config_file \
    --output_dir  $output_dir \
    --ckpt_path $ckpt_path_10 \
    --exec_step 8 \
    --task_suite_name  libero_10 \
    --device $gpu \
    --num_trails_per_task 50 \
    --task_ids 0 \
    --threshold 1


# EGL_DEVICE_ID=$gpu python  experiments/eval_libero.py \
#     --config_file $config_file \
#     --output_dir  $output_dir \
#     --ckpt_path $ckpt_path_obj \
#     --exec_step 8 \
#     --task_suite_name  libero_object \
#     --device $gpu \
#     --num_trails_per_task 50 \
#     --threshold 30

# EGL_DEVICE_ID=$gpu python  experiments/eval_libero.py \
#     --config_file $config_file \
#     --output_dir  $output_dir \
#     --ckpt_path $ckpt_path_spa \
#     --exec_step 8 \
#     --task_suite_name  libero_spatial \
#     --device $gpu \
#     --num_trails_per_task 50 \
#     --threshold 30