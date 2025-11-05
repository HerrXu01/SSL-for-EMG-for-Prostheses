python -u main.py pretrain \
  --config configs/phase_1_configs/phase_1_pretrain.yaml \
  --gpu 1 \
  --exp_name_prefix Phase_1_exp_6 \
  --seed 2026 \
  --split_filename phase_1_exp_6_sl80_tl16_mixFalse_splits.json \
  --enable_bp_filter \
  --seq_len 80 \
  --label_len 64 \
  --token_len 16 \
  --train_epochs 20 \
  --batch_size 64 \
  --learning_rate 0.0005
  