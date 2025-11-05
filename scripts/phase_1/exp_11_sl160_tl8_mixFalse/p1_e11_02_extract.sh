python -u main.py extract \
  --config configs/phase_1_configs/phase_1_extract.yaml \
  --gpu 1 \
  --seed 2026 \
  --seq_len 160 \
  --token_len 8 \
  --setting Phase_1_exp_11_AutoTimesLlama_Ninapro_DB5_epochs20_sl160_ll152_tl8_lr0.0005_bs64_hd512_hl2_mixFalse \
  --split_config_filename phase_1_exp_11_sl160_tl8_mixFalse_splits.json \
  --enable_bp_filter \
  --classifier_data_dir /mnt/scratchpad/xzt_ma/phase_1_exp_11_sl160_tl8_mixFalse/
