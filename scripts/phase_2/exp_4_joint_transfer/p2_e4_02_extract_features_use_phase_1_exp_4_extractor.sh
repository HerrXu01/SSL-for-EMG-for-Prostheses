python -u main.py extract \
  --config configs/phase_1_configs/phase_1_extract.yaml \
  --gpu 0 \
  --seed 2026 \
  --seq_len 80 \
  --token_len 4 \
  --setting Phase_1_exp_4_AutoTimesLlama_Ninapro_DB5_epochs20_sl80_ll76_tl4_lr0.0005_bs64_hd512_hl2_mixFalse \
  --split_config_filename phase_2_exp_4_diff_subjects_new_6gestures_splits.json \
  --enable_bp_filter \
  --classifier_data_dir /mnt/scratchpad/xzt_ma/phase_2_exp_4_extracted_features_use_phase_1_exp_4/
