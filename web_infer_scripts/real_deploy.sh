#!/usr/bin/bash


# IP_ADDRESS_OF_SERVER="0.0.0.0"
# DOMAIN_NAME="fold_towel"
# gpu_id=${1:-0}
# port=${2:-9001}
# action_dim=${3:-14}
# norm_type="minmax"
# python3 web_infer_utils/Real_deploy.py \
#     -c configs/ltx_model/fold_towel/action_model_WOV_WSTATE_absolute_real.yaml \
#     -w /defaultShare/Genie-Envisioner/lrz_training_v2/fold_towel/Phase3_PolicyModel_WOV_absolute_joint_WState/2026_03_19_16_28_36/step_40000/diffusion_pytorch_model.safetensors \
#     --denoise_step 10 \
#     --threshold 1 \
#     --host ${IP_ADDRESS_OF_SERVER} \
#     --port ${port} \
#     --domain_name ${DOMAIN_NAME} \
#     --action_dim ${action_dim} \
#     --norm_type ${norm_type} \
#     --device ${gpu_id}


IP_ADDRESS_OF_SERVER="0.0.0.0"
DOMAIN_NAME="fold_towel"
gpu_id=${1:-0}
port=${2:-9001}
action_dim=${3:-14}
norm_type="minmax"
python3 web_infer_utils/Real_deploy.py \
    -c configs/ltx_model/fold_towel/action_model_WV_WSTATE_absolute_real.yaml \
    -w /defaultShare/Genie-Envisioner/lrz_training_v2/fold_towel/Phase3_PolicyModel_WV_absolute_joint_WState/2026_03_27_21_35_58/step_40000/diffusion_pytorch_model.safetensors \
    --denoise_step 10 \
    --threshold 1 \
    --host ${IP_ADDRESS_OF_SERVER} \
    --port ${port} \
    --domain_name ${DOMAIN_NAME} \
    --action_dim ${action_dim} \
    --norm_type ${norm_type} \
    --device ${gpu_id}

# IP_ADDRESS_OF_SERVER="0.0.0.0"
# DOMAIN_NAME="fold_towel"
# gpu_id=${1:-0}
# port=${2:-9001}
# action_dim=${3:-14}
# norm_type="minmax"
# python3 web_infer_utils/Real_deploy.py \
#     -c configs/ltx_model/fold_towel/action_model_WV_WSTATE_absolute_real.yaml \
#     -w /defaultShare/Genie-Envisioner/lrz_training_v2/fold_towel/Phase3_PolicyModel_WV_absolute_joint_WState/2026_03_23_18_36_01/step_20000/diffusion_pytorch_model.safetensors \
#     --denoise_step 10 \
#     --threshold 1 \
#     --host ${IP_ADDRESS_OF_SERVER} \
#     --port ${port} \
#     --domain_name ${DOMAIN_NAME} \
#     --action_dim ${action_dim} \
#     --norm_type ${norm_type} \
#     --device ${gpu_id}


# IP_ADDRESS_OF_SERVER="0.0.0.0"
# DOMAIN_NAME="orangize_bowls_cups"
# gpu_id=${1:-0}
# port=${2:-9001}
# action_dim=${3:-14}
# norm_type="minmax"
# python3 web_infer_utils/Real_deploy.py \
#     -c configs/ltx_model/orangize_bowls_cups/action_model_WOV_WSTATE_absolute_real.yaml \
#     -w /defaultShare/Genie-Envisioner/lrz_training_v2/bowls_cups/Phase3_PolicyModel_WOV_absolute_joint_WState/2026_03_06_11_36_35/step_10000/diffusion_pytorch_model.safetensors \
#     --denoise_step 10 \
#     --threshold 1 \
#     --host ${IP_ADDRESS_OF_SERVER} \
#     --port ${port} \
#     --domain_name ${DOMAIN_NAME} \
#     --action_dim ${action_dim} \
#     --norm_type ${norm_type} \
#     --device ${gpu_id}



# IP_ADDRESS_OF_SERVER="0.0.0.0"
# DOMAIN_NAME="orangize_bowls_cups"
# gpu_id=${1:-0}
# port=${2:-9001}
# action_dim=${3:-14}
# norm_type="minmax"
# python3 web_infer_utils/Real_deploy.py \
#     -c configs/ltx_model/orangize_bowls_cups/action_model_WV_WSTATE_absolute_real.yaml \
#     -w /defaultShare/Genie-Envisioner/lrz_training_v2/bowls_cups/Phase3_PolicyModel_WV_absolute_joint_WState/2026_03_06_21_21_09/step_10000/diffusion_pytorch_model.safetensors \
#     --denoise_step 10 \
#     --threshold 1 \
#     --host ${IP_ADDRESS_OF_SERVER} \
#     --port ${port} \
#     --domain_name ${DOMAIN_NAME} \
#     --action_dim ${action_dim} \
#     --norm_type ${norm_type} \
#     --device ${gpu_id}







# IP_ADDRESS_OF_SERVER="0.0.0.0"
# DOMAIN_NAME="place_3"
# gpu_id=${1:-0}
# port=${2:-9001}
# action_dim=${3:-14}
# norm_type="minmax"
# python3 web_infer_utils/Real_deploy.py \
#     -c configs/ltx_model/place_3/action_model_WOV_WSTATE_absolute_real.yaml \
#     -w lrz_training_v2/place3/Phase3_PolicyModel_WOV_absolute_joint_WState/2026_03_12_21_37_34/step_20000/diffusion_pytorch_model.safetensors \
#     --denoise_step 10 \
#     --threshold 1 \
#     --host ${IP_ADDRESS_OF_SERVER} \
#     --port ${port} \
#     --domain_name ${DOMAIN_NAME} \
#     --action_dim ${action_dim} \
#     --norm_type ${norm_type} \
#     --device ${gpu_id}


# IP_ADDRESS_OF_SERVER="0.0.0.0"
# DOMAIN_NAME="place_3"
# gpu_id=${1:-0}
# port=${2:-9001}
# action_dim=${3:-14}
# norm_type="minmax"
# python3 web_infer_utils/Real_deploy.py \
#     -c configs/ltx_model/place_3/action_model_WV_WSTATE_absolute_real.yaml \
#     -w lrz_training_v2/place3/Phase3_PolicyModel_WV_absolute_joint_WState/2026_03_14_10_45_25/step_21000/diffusion_pytorch_model.safetensors \
#     --denoise_step 10 \
#     --threshold 1 \
#     --host ${IP_ADDRESS_OF_SERVER} \
#     --port ${port} \
#     --domain_name ${DOMAIN_NAME} \
#     --action_dim ${action_dim} \
#     --norm_type ${norm_type} \
#     --device ${gpu_id}