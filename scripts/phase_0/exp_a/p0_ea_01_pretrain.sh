python -u main.py pretrain \
  --config configs/phase_0_configs/phase_0_pretrain.yaml \
  --gpu 0 \
  --exp_name_prefix Phase_0_exp_a_no_bp_filter_no_rectify \
  --seed 2025 \
  --split_filename phase_0_exp_a_no_bp_filter_no_rectify_splits.json \
  --seq_len 160 \
  --label_len 152 \
  --token_len 8 \
  --mix_embeds \
  --train_epochs 20 \
  --batch_size 64 \
  --learning_rate 0.0005
  