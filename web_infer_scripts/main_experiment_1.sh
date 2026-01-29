#!/usr/bin/bash

IP_ADDRESS_OF_SERVER="127.0.0.1"
DOMAIN_NAME="robotwin_clean_full"

for i in {0..3}; do
    gpu_id=$i
    port=$((8001 + i))
    python3 web_infer_scripts/main_server.py \
        -c configs/ltx_model/robotwin/policy_model_server_eef.yaml \
        -w lrz_training_v2/robotwin_full_clean/Phase3_PolicyModel_WOV_absolute_eef/2026_01_13_21_12_10/step_30000/diffusion_pytorch_model.safetensors \
        --denoise_step 10 \
        --threshold 1 \
        --host ${IP_ADDRESS_OF_SERVER} \
        --port ${port} \
        --domain_name ${DOMAIN_NAME} \
        --device ${gpu_id} &
    echo "Started server on GPU ${gpu_id} with port ${port}"
done

# 等待所有后台进程
wait
