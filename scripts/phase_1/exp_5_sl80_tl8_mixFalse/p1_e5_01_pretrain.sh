python -u main.py pretrain \
  --config configs/phase_1_configs/phase_1_pretrain.yaml \
  --gpu 0 \
  --exp_name_prefix Phase_1_exp_5 \
  --seed 2026 \
  --split_filename phase_1_exp_5_sl80_tl8_mixFalse_splits.json \
  --enable_bp_filter \
  --seq_len 80 \
  --label_len 72 \
  --token_len 8 \
  --train_epochs 20 \
  --batch_size 64 \
  --learning_rate 0.0005
  