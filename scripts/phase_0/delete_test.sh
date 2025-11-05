python -u main.py pretrain \
  --config configs/phase_0_configs/phase_0_pretrain.yaml \
  --gpu 0 \
  --exp_name_prefix delete_this_test_speed \
  --seed 2025 \
  --split_filename delete_this_test_speed.json \
  --seq_len 160 \
  --label_len 152 \
  --token_len 8 \
  --mix_embeds \
  --train_epochs 20 \
  --batch_size 128 \
  --learning_rate 0.0005
  