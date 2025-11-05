python -u main.py extract \
  --config configs/phase_1_configs/phase_1_extract.yaml \
  --gpu 1 \
  --seed 2026 \
  --seq_len 80 \
  --token_len 16 \
  --setting Phase_1_exp_6_AutoTimesLlama_Ninapro_DB5_epochs20_sl80_ll64_tl16_lr0.0005_bs64_hd512_hl2_mixFalse \
  --split_config_filename phase_1_exp_6_sl80_tl16_mixFalse_splits.json \
  --enable_bp_filter \
  --classifier_data_dir /mnt/scratchpad/xzt_ma/phase_1_exp_6_sl80_tl16_mixFalse/
