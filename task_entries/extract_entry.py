import os
import torch
import json
import random
import numpy as np
from tqdm import tqdm
from feature_learners.AutoTimes_Llama import AutoTimesLlama
from utils.build_h5_dataset import init_h5_file, write_batch_to_h5
from utils.window_preprocess import window_preprocess
from utils.tools import load_and_merge_config
from utils.registry import registry


def extract_features(window, args, timestamp_embedding, model):
    filtered_window = np.concatenate([
        window_preprocess(
            window[:, i:i+1],
            enable_bp_filter=args.enable_bp_filter,
            enable_rectify=args.enable_rectify,
            enable_lp_filter=args.enable_lp_filter,
            cutoff_f=args.cutoff_f
        ) for i in range(window.shape[1])
    ], axis=1)

    window_tensor = torch.tensor(filtered_window, dtype=torch.float16).T.unsqueeze(-1).to(model.device)  # shape: (8, seq_len, 1)
    timestamp_embedding = timestamp_embedding.float().to(model.device)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            feature = model(window_tensor, timestamp_embedding).cpu().numpy()  # (8, num_tokens, 4096)
            feature = feature[:, -1, :].T  # shape: (4096, 8)------------------------------------------------> new
    return feature

def save_windows_to_h5(current_data, current_labels, args, subject, gesture, flag, gesture_label_map,
                        timestamp_embedding, feature_extractor, ssl_h5, supervised_h5):
    features_buffer = []
    rawdata_buffer = []
    labels_buffer = []
    window_range = range(0, current_data.shape[0] - args.seq_len + 1)
    for i in tqdm(window_range, desc=f"subject {subject}, gesture {gesture}, {flag}"):
        window_data = current_data[i:i+args.seq_len]  # Shape (seq_len, 8)
        window_label = current_labels[i:i+args.seq_len]

        binary_arr = (window_label != 0).astype(np.float16)
        vote = binary_arr.mean()
        if vote >= args.label_threshold:
            current_label = gesture_label_map[gesture]
        else:
            current_label = 0  # Rest

        feature = extract_features(window_data, args, timestamp_embedding, feature_extractor)
        features_buffer.append(feature.astype(np.float16))
        rawdata_buffer.append(window_data.astype(np.float16))
        labels_buffer.append(current_label)

    write_batch_to_h5(ssl_h5, features_buffer, labels_buffer)
    write_batch_to_h5(supervised_h5, rawdata_buffer, labels_buffer)


@registry.register_task("extract")
def run_extract(args):
    args = load_and_merge_config(args)
    device = torch.device(f"cuda:{args.gpu}")
    os.makedirs(args.classifier_data_dir, exist_ok=True)

    feature_extractor = AutoTimesLlama(args, mode='extract').to(device)
    ckpt_path = os.path.join(args.checkpoints, args.setting, args.ckpt_filename)
    ckpt = torch.load(ckpt_path, map_location=device)
    filtered_state = {k: v for k, v in ckpt.items() if k.startswith('encoder') or k == 'add_scale'}
    missing, unexpected = feature_extractor.load_state_dict(filtered_state, strict=False)
    print(f"Loaded encoder & add_scale with missing keys: {missing}, unexpected keys: {unexpected}")
    feature_extractor.eval()

    token_num = args.seq_len // args.token_len

    """
    ssl_train_h5 = init_h5_file(os.path.join(args.classifier_data_dir, "ssl_train.h5"), (8, token_num, 4096))
    ssl_val_h5 = init_h5_file(os.path.join(args.classifier_data_dir, "ssl_val.h5"), (8, token_num, 4096))
    ssl_test_h5 = init_h5_file(os.path.join(args.classifier_data_dir, "ssl_test.h5"), (8, token_num, 4096))
    """
    ssl_train_h5 = init_h5_file(os.path.join(args.classifier_data_dir, "ssl_train.h5"), (4096, 8))
    ssl_val_h5 = init_h5_file(os.path.join(args.classifier_data_dir, "ssl_val.h5"), (4096, 8))
    ssl_test_h5 = init_h5_file(os.path.join(args.classifier_data_dir, "ssl_test.h5"), (4096, 8))
    supervised_train_h5 = init_h5_file(os.path.join(args.classifier_data_dir, "supervised_train.h5"), (args.seq_len, 8))
    supervised_val_h5 = init_h5_file(os.path.join(args.classifier_data_dir, "supervised_val.h5"), (args.seq_len, 8))
    supervised_test_h5 = init_h5_file(os.path.join(args.classifier_data_dir, "supervised_test.h5"), (args.seq_len, 8))

    timestamp_embedding_filename = f"sl{args.seq_len}_tl{args.token_len}_sr200.pt"
    timestamp_embedding_path = os.path.join(args.timestamp_embedding_dir, timestamp_embedding_filename)
    timestamp_embedding = torch.load(timestamp_embedding_path)  # shape (num_tokens, 4096)

    split_config_path = os.path.join(args.split_config_dir, args.split_config_filename)
    with open(split_config_path, "r", encoding="utf-8") as f:
        repeat_splits = json.load(f)

    any_sbj_dict = next(iter(repeat_splits.values()))
    gesture_list = list(any_sbj_dict.keys())
    gesture_label_map = {g: lb for lb, g in enumerate(gesture_list, start=1)}
    gesture_label_map["Rest"] = 0
    print(f"gesture to label map: {gesture_label_map} \n")
    
    for subject, sbj_dict in repeat_splits.items():
        for gesture, splits in sbj_dict.items():
            current_train_data = np.array(splits["classifier_train_data"])  # Shape (T, 8)
            current_train_labels = np.array(splits["classifier_train_restimulus"])  # Shape (T,)
            
            save_windows_to_h5(
                current_data=current_train_data,
                current_labels=current_train_labels,
                args=args,
                subject=subject,
                gesture=gesture,
                flag="train data",
                gesture_label_map=gesture_label_map,
                timestamp_embedding=timestamp_embedding,
                feature_extractor=feature_extractor,
                ssl_h5=ssl_train_h5,
                supervised_h5=supervised_train_h5
            )
            
            current_valtest_data = np.array(splits["classifier_valtest_data"])
            current_valtest_labels = np.array(splits["classifier_valtest_restimulus"])

            cut_point = current_valtest_data.shape[0] // 2
            first_half_data = current_valtest_data[:cut_point]
            second_half_data = current_valtest_data[cut_point:]
            first_half_labels = current_valtest_labels[:cut_point]
            second_half_labels = current_valtest_labels[cut_point:]

            if random.random() < 0.5:
                current_val_data = first_half_data
                current_val_labels = first_half_labels
                current_test_data = second_half_data
                current_test_labels = second_half_labels
            else:
                current_val_data = second_half_data
                current_val_labels = second_half_labels
                current_test_data = first_half_data
                current_test_labels = first_half_labels

            save_windows_to_h5(
                current_data=current_val_data,
                current_labels=current_val_labels,
                args=args,
                subject=subject,
                gesture=gesture,
                flag="val data",
                gesture_label_map=gesture_label_map,
                timestamp_embedding=timestamp_embedding,
                feature_extractor=feature_extractor,
                ssl_h5=ssl_val_h5,
                supervised_h5=supervised_val_h5
            )

            save_windows_to_h5(
                current_data=current_test_data,
                current_labels=current_test_labels,
                args=args,
                subject=subject,
                gesture=gesture,
                flag="test data",
                gesture_label_map=gesture_label_map,
                timestamp_embedding=timestamp_embedding,
                feature_extractor=feature_extractor,
                ssl_h5=ssl_test_h5,
                supervised_h5=supervised_test_h5
            )

    label_map_path = os.path.join(args.classifier_data_dir, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(gesture_label_map, f, indent=2)

    print(f"\nDone. SSL data and supervised data and label map are saved to {args.classifier_data_dir}")
