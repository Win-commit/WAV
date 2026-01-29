#!/usr/bin/bash

IP_ADDRESS_OF_SERVER="127.0.0.1"
DOMAIN_NAME="robotwin"

# 启动8个服务器，使用GPU 0-7，端口8001-8008
for i in {0..7}; do
    gpu_id=$i
    port=$((8001 + i))
    python3 web_infer_scripts/main_server.py \
        -c configs/ltx_model/robotwin/policy_model_server_eef.yaml \
        -w lrz_training_v2/Phase3_PolicyModel_Wvalue_EEF_v2/2025_12_27_02_07_47/step_20000/diffusion_pytorch_model.safetensors \
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
