python -u main.py pretrain \
  --config configs/phase_2_configs/phase_2_exp_additional_subject_transfer.yaml \
  --gpu 0 \
  --exp_name_prefix Phase_2_exp_additional_subject_transfer_to_get_split_config \
  --seed 2026 \
  --split_filename phase_2_exp_additional_subject_transfer.json \
  --enable_bp_filter \
  --seq_len 80 \
  --label_len 76 \
  --token_len 4 \
  --train_epochs 20 \
  --batch_size 64 \
  --learning_rate 0.0005
  