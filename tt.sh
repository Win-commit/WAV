bash scripts/infer.sh main.py \
    /defaultShare/Genie-Envisioner/configs/ltx_model/robotwin/test_policy_eef.yaml \
   /defaultShare/Genie-Envisioner/lrz_training_v2/robotwin_full_clean/Phase3_PolicyModel_WOV_absolute_eef/2026_01_13_21_12_10/step_30000/diffusion_pytorch_model.safetensors \
    /defaultShare/Genie-Envisioner/value_vis/robotwin_clean_full/absolute_eef_action_WOV \
    robotwin_clean_full

# python scripts/get_statistics.py --data_root /liujinxin/zhy/lirunze/Genie-Envisioner/lerobotlike_datasets/robotwin/data --data_name robotwin --data_type joint --action_key action --state_key observation.state --value_key state_value --save_path /liujinxin/zhy/lirunze/Genie-Envisioner/lerobotlike_datasets/robotwin/meta/stats.json

# bash scripts/infer.sh main.py \
#     /defaultShare/Genie-Envisioner/configs/ltx_model/robotwin/test_video.yaml \
#    /defaultShare/Genie-Envisioner/lirunze_training_v2/robotwin_full_clean/Phase1_VideoAdapter/2026_01_11_16_23_03/step_55000/diffusion_pytorch_model.safetensors \
#     /defaultShare/Genie-Envisioner/value_vis/robotwin_clean_full/video \
#     robotwin_clean_full

# bash scripts/infer.sh main.py \
#     /defaultShare/Genie-Envisioner/configs/ltx_model/robotwin/test_value.yaml \
#    /defaultShare/Genie-Envisioner/lrz_training_v2/robotwin_full_clean/Phase2_ValueModel/2026_01_13_21_09_23/step_50000/diffusion_pytorch_model.safetensors \
#     /defaultShare/Genie-Envisioner/value_vis/robotwin_clean_full/value \
#     robotwin_clean_full