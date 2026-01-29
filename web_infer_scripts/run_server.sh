#!/usr/bin/bash

IP_ADDRESS_OF_SERVER="127.0.0.1"
DOMAIN_NAME="robotwin"
gpu_id=${1:-0}
port=${2:-8001}
python3 web_infer_scripts/main_server.py \
    -c configs/ltx_model/robotwin/policy_model_server_eef.yaml \
    -w lrz_training_v2/Phase3_PolicyModel_Wvalue_EEF_v2/2025_12_27_02_07_47/step_20000/diffusion_pytorch_model.safetensors \
    --denoise_step 10 \
    --threshold 1 \
    --host ${IP_ADDRESS_OF_SERVER} \
    --port ${port} \
    --domain_name ${DOMAIN_NAME} \
    --device ${gpu_id}

# python3 web_infer_scripts/main_server.py \
#     -c configs/ltx_model/robotwin/policy_model_server.yaml \
#     -w /liujinxin/zhy/lirunze/Genie-Envisioner/lrz_training_v2/Phase3_PolicyModel_Wvalue_v2/2025_12_24_11_24_03/step_10000/diffusion_pytorch_model.safetensors \
#     --denoise_step 10 \
#     --threshold 1 \
#     --host ${IP_ADDRESS_OF_SERVER} \
#     --port 8102 \
#     --domain_name ${DOMAIN_NAME} \
#     --device 2 &


# python3 web_infer_scripts/main_server.py \
#     -c configs/ltx_model/robotwin/policy_model_server.yaml \
#     -w /liujinxin/zhy/lirunze/Genie-Envisioner/lrz_training_v2/Phase3_PolicyModel_Wvalue_v2/2025_12_24_11_24_03/step_10000/diffusion_pytorch_model.safetensors \
#     --denoise_step 10 \
#     --threshold 1 \
#     --host ${IP_ADDRESS_OF_SERVER} \
#     --port 8103 \
#     --domain_name ${DOMAIN_NAME} \
#     --device 2 &

# python3 web_infer_scripts/main_server.py \
#     -c configs/ltx_model/robotwin/policy_model_server.yaml \
#     -w /liujinxin/zhy/lirunze/Genie-Envisioner/lrz_training_v2/Phase3_PolicyModel_Wvalue_v2/2025_12_24_11_24_03/step_10000/diffusion_pytorch_model.safetensors \
#     --denoise_step 10 \
#     --threshold 1 \
#     --host ${IP_ADDRESS_OF_SERVER} \
#     --port 8104 \
#     --domain_name ${DOMAIN_NAME} \
#     --device 1 &

# wait
