# bash scripts/infer.sh main.py \
#     /defaultShare/Genie-Envisioner/configs/ltx_model/place_3/value_model_real.yaml \
#    /defaultShare/Genie-Envisioner/lrz_training_v2/place3/Phase2_ValueModel/2026_03_12_21_38_21/step_20000/diffusion_pytorch_model.safetensors \
#     /defaultShare/Genie-Envisioner/value_vis/place_3/value/3 \
#     place_3

# bash scripts/infer.sh main.py \
#     /defaultShare/Genie-Envisioner/configs/ltx_model/place_3/action_model_WV_WSTATE_absolute_real.yaml \
#    /defaultShare/Genie-Envisioner/lrz_training_v2/place3/Phase3_PolicyModel_WV_absolute_joint_WState/2026_03_14_10_45_25/step_21000/diffusion_pytorch_model.safetensors \
#     /defaultShare/Genie-Envisioner/value_vis/place_3/abso_action/8/WV \
#     place_3

# bash scripts/infer.sh main.py \
#     /defaultShare/Genie-Envisioner/configs/ltx_model/place_3/action_model_WV_WSTATE_absolute_real.yaml \
#    /defaultShare/Genie-Envisioner/lrz_training_v2/place3/Phase3_PolicyModel_WV_absolute_joint_WState/2026_03_14_10_45_25/step_9000/diffusion_pytorch_model.safetensors \
#     /defaultShare/Genie-Envisioner/value_vis/place_3/abso_action/8/WV \
#     place_3


bash scripts/infer.sh main.py \
    /defaultShare/Genie-Envisioner/configs/ltx_model/orangize_bowls_cups/video_model_real.yaml \
   /defaultShare/Genie-Envisioner/lrz_training_v2/bowls_cups/Phase1_VideoAdapter/2026_03_02_12_52_13/step_40000/diffusion_pytorch_model.safetensors \
    /defaultShare/Genie-Envisioner/value_vis/orangize_bowls_cups/video \
    orangize_bowls_cups

# bash scripts/infer.sh main.py \
#     /defaultShare/Genie-Envisioner/configs/ltx_model/orangize_bowls_cups/action_model_WV_WSTATE_absolute_real.yaml \
#    /defaultShare/Genie-Envisioner/lrz_training_v2/bowls_cups/Phase3_PolicyModel_WV_absolute_joint_WState/2026_03_06_21_21_09/step_30000/diffusion_pytorch_model.safetensors \
#     /defaultShare/Genie-Envisioner/value_vis/orangize_bowls_cups/absolute_action \
#     orangize_bowls_cups


# bash scripts/infer.sh main.py \
#     /defaultShare/Genie-Envisioner/configs/ltx_model/fold_towel/video_model_real.yaml \
#    /defaultShare/Genie-Envisioner/lrz_training_v2/fold_towel/Phase1_VideoAdapter/2026_03_17_12_33_53/step_40000/diffusion_pytorch_model.safetensors \
#     /defaultShare/Genie-Envisioner/value_vis/fold_towel_big/video/3 \
#     fold_towel


# bash scripts/infer.sh main.py \
#     /defaultShare/Genie-Envisioner/configs/ltx_model/fold_towel/action_model_WV_WSTATE_absolute_real.yaml \
#    /defaultShare/Genie-Envisioner/lrz_training_v2/fold_towel/Phase3_PolicyModel_WV_absolute_joint_WState/2026_03_27_21_35_58/step_60000/diffusion_pytorch_model.safetensors \
#     /defaultShare/Genie-Envisioner/value_vis/fold_towel_big/absolute_action/3/WV \
#     fold_towel

# bash scripts/infer.sh main.py \
#     /defaultShare/Genie-Envisioner/configs/ltx_model/fold_towel/validation.yaml \
#    /defaultShare/Genie-Envisioner/lrz_training_v2/fold_towel/Phase3_PolicyModel_WV_absolute_joint_WState/2026_03_27_21_35_58/step_60000/diffusion_pytorch_model.safetensors\
#     /defaultShare/Genie-Envisioner/value_vis/fold_towel_big/absolute_action/validation/3/WV \
#     fold_towel_test

# bash scripts/infer.sh main.py \
#     /defaultShare/Genie-Envisioner/configs/ltx_model/fold_towel/validation.yaml \
#    /defaultShare/Genie-Envisioner/lrz_training_v2/fold_towel/antiOverfit/Phase3_PolicyModel_WOV_absolute_joint_WState_v2/2026_03_12_13_44_00/step_10000/diffusion_pytorch_model.safetensors \
#     /defaultShare/Genie-Envisioner/value_vis/fold_towel/absolute_action \
#     fold_towel_test

# bash scripts/infer.sh main.py \
#     /defaultShare/Genie-Envisioner/configs/ltx_model/fold_towel/validation.yaml \
#    /defaultShare/Genie-Envisioner/lrz_training_v2/fold_towel/antiOverfit/Phase3_PolicyModel_WOV_absolute_joint_WState_v2/2026_03_12_13_44_00/step_12000/diffusion_pytorch_model.safetensors \
#     /defaultShare/Genie-Envisioner/value_vis/fold_towel/absolute_action \
#     fold_towel_test


# bash scripts/infer.sh main.py \
#     /defaultShare/Genie-Envisioner/configs/ltx_model/fold_towel/action_model_WOV_WSTATE_relative_real.yaml \
#    /defaultShare/Genie-Envisioner/lrz_training_v2/fold_towel/Phase3_PolicyModel_WOV_relative_joint_WState/2026_03_07_18_40_56/step_40000/diffusion_pytorch_model.safetensors \
#     /defaultShare/Genie-Envisioner/value_vis/fold_towel/relative_action \
#     fold_towel

# bash scripts/infer.sh main.py \
#     /defaultShare/Genie-Envisioner/configs/ltx_model/libero/video_model_libero.yaml \
#    /defaultShare/Genie-Envisioner/lrz_training_v2/libero/Phase1_VideoAdapter/2026_02_07_17_38_14/step_30000/diffusion_pytorch_model.safetensors \
#     /defaultShare/Genie-Envisioner/value_vis/libero/video/ \
#     libero

# bash scripts/infer.sh main.py \
#     /defaultShare/Genie-Envisioner/configs/ltx_model/robotwin/policy_model_rel_joint_robotwin_add_state.yaml \
#    /defaultShare/Genie-Envisioner/lrz_training_v2/robotwin4/Phase3_PolicyModel_WOV_rel_joint_STATE_54/2026_03_02_14_36_43/step_20000/diffusion_pytorch_model.safetensors \
#     /defaultShare/Genie-Envisioner/value_vis/robotwin4/action/new_process \
#     robotwin

# bash scripts/infer.sh main.py \
#     /defaultShare/Genie-Envisioner/configs/ltx_model/robotwin/policy_model_rel_joint_robotwin.yaml \
#    /defaultShare/Genie-Envisioner/lrz_training_v2/robotwin4/Phase3_PolicyModel_WOV_rel_joint_54/2026_02_22_19_12_11/step_20000/diffusion_pytorch_model.safetensors \
#     /defaultShare/Genie-Envisioner/value_vis/robotwin4/action/new_process \
#     robotwin

# python scripts/get_statistics.py --data_root /liujinxin/zhy/lirunze/Genie-Envisioner/lerobotlike_datasets/robotwin/data --data_name robotwin --data_type joint --action_key action --state_key observation.state --value_key state_value --save_path /liujinxin/zhy/lirunze/Genie-Envisioner/lerobotlike_datasets/robotwin/meta/stats.json



# bash scripts/infer.sh main.py \
#     /defaultShare/Genie-Envisioner/configs/ltx_model/calvin/video_model_calvin.yaml \
#    /defaultShare/Genie-Envisioner/lrz_training_v2/calvin/Phase1_VideoAdapter/2026_03_20_11_22_26/step_30000/diffusion_pytorch_model.safetensors \
#     /defaultShare/Genie-Envisioner/value_vis/calvin/video/3 \
#     calvin_ABC_D

# bash scripts/infer.sh main.py \
#     /defaultShare/Genie-Envisioner/configs/ltx_model/calvin/action_model_calvin_abso.yaml \
#    /defaultShare/Genie-Envisioner/lrz_training_v2/calvin/Phase3_PolicyModel_WOV_rel_WState/2026_03_30_15_44_33/step_50000/diffusion_pytorch_model.safetensors \
#     /defaultShare/Genie-Envisioner/value_vis/calvin/abso_action/3/WOV \
#     calvin_ABC_D

# bash scripts/infer.sh main.py \
#     /defaultShare/Genie-Envisioner/configs/ltx_model/calvin/value_model_calvin.yaml \
#    /defaultShare/Genie-Envisioner/lrz_training_v2/calvin/Phase2_ValueModel/2026_03_23_18_40_06/step_60000/diffusion_pytorch_model.safetensors \
#     /defaultShare/Genie-Envisioner/value_vis/calvin/value/3 \
#     calvin_ABC_D