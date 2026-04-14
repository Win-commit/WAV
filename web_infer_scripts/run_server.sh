#!/usr/bin/bash

IP_ADDRESS_OF_SERVER="127.0.0.1"
DOMAIN_NAME="Test"
gpu_id=${1:-0}
port=${2:-8001}
action_dim=${3:-14}
norm_type=${4:"meanstd"} #[meanstd,minmax] 
python3 web_infer_scripts/main_server.py \
    -c configs/ltx_model/PathToYourConfig \
    -w Path/To/Your/Weight \
    --denoise_step 10 \
    --threshold 1 \
    --host ${IP_ADDRESS_OF_SERVER} \
    --port ${port} \
    --domain_name ${DOMAIN_NAME} \
    --action_dim ${action_dim} \
    --norm_type ${norm_type}  \
    --device ${gpu_id}

