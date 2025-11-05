python -u main.py pretrain \
  --config configs/phase_1_configs/phase_1_pretrain.yaml \
  --gpu 1 \
  --exp_name_prefix Phase_1_exp_12 \
  --seed 2026 \
  --split_filename phase_1_exp_12_sl160_tl16_mixFalse_splits.json \
  --enable_bp_filter \
  --seq_len 160 \
  --label_len 144 \
  --token_len 16 \
  --train_epochs 20 \
  --batch_size 64 \
  --learning_rate 0.0005
  