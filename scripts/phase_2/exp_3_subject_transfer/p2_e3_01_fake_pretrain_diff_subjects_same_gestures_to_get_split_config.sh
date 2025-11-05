python -u main.py pretrain \
  --config configs/phase_2_configs/phase_2_exp_3_diff_subjects_same_gestures_fake_pretrain.yaml \
  --gpu 0 \
  --exp_name_prefix Phase_2_exp_3_fake_pretrain_diff_subjects_same_gestures_to_get_split_config \
  --seed 2026 \
  --split_filename phase_2_exp_3_diff_subjects_old_6gestures_splits.json \
  --enable_bp_filter \
  --seq_len 80 \
  --label_len 76 \
  --token_len 4 \
  --train_epochs 20 \
  --batch_size 64 \
  --learning_rate 0.0005
  