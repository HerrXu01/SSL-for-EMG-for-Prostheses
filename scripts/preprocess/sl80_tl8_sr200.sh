python -u main.py preprocess \
  --config configs/preprocess_base.yaml \
  --gpu 0 \
  --seq_len 80 \
  --token_len 8 \
  --sampling_rate 200 \
  --timestamp_embedding_path dataset/timestamp_embeddings
