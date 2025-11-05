from utils.llama_time_encoder import LlamaTimeEncoder
from datetime import timedelta, datetime
import torch
import os
from utils.tools import load_and_merge_config
from utils.registry import registry


@registry.register_task("preprocess")
def run_preprocess(args):
    args = load_and_merge_config(args)
    model = LlamaTimeEncoder(args)
    model.eval()

    assert args.seq_len % args.token_len == 0, "seq_len must be divisible by token_len"
    num_tokens = args.seq_len // args.token_len
    time_interval = 1.0 / args.sampling_rate

    sentences = []
    for i in range(num_tokens): 
        start_sec = i * args.token_len * time_interval
        end_sec = start_sec + (args.token_len - 1) * time_interval

        start_time = (datetime.min + timedelta(seconds=start_sec)).strftime("%H:%M:%S.%f")[:-3]
        end_time = (datetime.min + timedelta(seconds=end_sec)).strftime("%H:%M:%S.%f")[:-3]

        sentence = f"This is Time Series sampled at {args.sampling_rate}Hz, spanning from {start_time} to {end_time}."
        sentences.append(sentence)

    with torch.no_grad():
        timestamp_embeddings = model(sentences)
    
    filename = f"sl{args.seq_len}_tl{args.token_len}_sr{args.sampling_rate}.pt"
    os.makedirs(args.timestamp_embedding_path, exist_ok=True)
    save_path = os.path.join(args.timestamp_embedding_path, filename)

    torch.save(timestamp_embeddings, save_path)
    print(f"Timestamp embeddings saved to {save_path}")