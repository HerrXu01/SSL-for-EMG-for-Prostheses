python -u main.py pretrain \
  --config configs/phase_1_configs/phase_1_pretrain.yaml \
  --gpu 0 \
  --exp_name_prefix Phase_1_exp_15 \
  --seed 2026 \
  --split_filename phase_1_exp_15_sl320_tl16_mixTrue_splits.json \
  --enable_bp_filter \
  --seq_len 320 \
  --label_len 304 \
  --token_len 16 \
  --mix_embeds \
  --train_epochs 20 \
  --batch_size 64 \
  --learning_rate 0.0005
  