#!/usr/bin/bash

IP_ADDRESS_OF_SERVER="127.0.0.1"
DOMAIN_NAME="robotwin"
gpu_id=${1:-0}
port=${2:-8001}
python3 web_infer_scripts/main_server.py \
    -c configs/ltx_model/robotwin/policy_model_server_joint.yaml \
    -w lrz_training_v2/Phase3_PolicyModel_WOV_absolute_joint/2026_01_09_09_47_11/step_50000/diffusion_pytorch_model.safetensors \
    --denoise_step 10 \
    --threshold 1 \
    --host ${IP_ADDRESS_OF_SERVER} \
    --port ${port} \
    --domain_name ${DOMAIN_NAME} \
    --device ${gpu_id}

