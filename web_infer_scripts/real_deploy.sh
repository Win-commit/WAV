#!/usr/bin/bash

IP_ADDRESS_OF_SERVER="0.0.0.0"
DOMAIN_NAME="DATASET_NAME"
gpu_id=${1:-0}
port=${2:-9001}
action_dim=${3:-14}
norm_type="minmax"
python3 web_infer_utils/Real_deploy.py \
    -c CONFIG_PATH \
    -w WEIGHT_PATH \
    --denoise_step 10 \
    --threshold 1 \
    --host ${IP_ADDRESS_OF_SERVER} \
    --port ${port} \
    --domain_name ${DOMAIN_NAME} \
    --action_dim ${action_dim} \
    --norm_type ${norm_type} \
    --device ${gpu_id}

