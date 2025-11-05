import argparse
import random
import numpy as np
import torch
from task_entries.pretrain_entry import run_pretrain
from task_entries.preprocess_entry import run_preprocess
from task_entries.extract_entry import run_extract
from task_entries.classifier_entry import run_classifier
from utils.registry import registry


def main():
    parser = argparse.ArgumentParser(description='Self-Supervised Learning for EMG: Main Entry')
    subparsers = parser.add_subparsers(dest="task", required=True)

    # Subparser: preprocess
    preprocess_parser = subparsers.add_parser("preprocess")
    preprocess_parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file for preprocessing')
    preprocess_parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    preprocess_parser.add_argument('--seq_len', type=int, default=80, help='input sequence/window length')
    preprocess_parser.add_argument('--token_len', type=int, default=4, help='token length, how many sequence data points form a token')
    preprocess_parser.add_argument('--sampling_rate', type=int, default=200, help='The sampling rate of the data')
    preprocess_parser.add_argument('--timestamp_embedding_path', type=str, default='dataset', help='The folder name for saving the pre-calculated timestamp embeddings')
    preprocess_parser.add_argument('--seed', type=int, default=2025, help='random seed')
    
    # Subparser: pretrain
    pretrain_parser = subparsers.add_parser("pretrain")
    pretrain_parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file for pretraining.')
    pretrain_parser.add_argument('--gpu', type=int, default=0, help='gpu')
    pretrain_parser.add_argument('--exp_name_prefix', type=str, required=True, help='The initial part of the experiment name')
    pretrain_parser.add_argument('--seed', type=int, default=2025, help='random seed')
    pretrain_parser.add_argument('--split_filename', type=str, default='Phase_0_splits.json', help='The filename of how we split the repeats for pretrain and classifier')
    pretrain_parser.add_argument('--enable_bp_filter', action='store_true',  default=False, help='Enable bandpass filter for preprocessing EMG')
    pretrain_parser.add_argument('--enable_rectify', action='store_true',  default=False, help='Enable rectification for preprocessing EMG')
    pretrain_parser.add_argument('--enable_lp_filter', action='store_true',  default=False, help='Enable lowpass filter for preprocessing EMG')
    pretrain_parser.add_argument('--cutoff_f', type=int, default=90, help='The cutoff frequency of the lowpass filter (in enabled)')
    pretrain_parser.add_argument('--seq_len', type=int, default=80, help='The sequence length of the input')
    pretrain_parser.add_argument('--label_len', type=int, default=76, help='The sequence length of the label')
    pretrain_parser.add_argument('--token_len', type=int, default=4, help='How many time series data points we consider as a token')
    pretrain_parser.add_argument('--mix_embeds', action='store_true',  default=False, help='Add timestamp embedding to the inputs for the llama')
    pretrain_parser.add_argument('--train_epochs', type=int, default=10, help='Epochs for pretraining')
    pretrain_parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    pretrain_parser.add_argument('--learning_rate', type=float, default=0.0005, help='optimizer learning rate')

    # Subparser: extract
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file for feature extraction (build ssl dataset).')
    extract_parser.add_argument('--gpu', type=int, default=0, help='gpu')
    extract_parser.add_argument('--seed', type=int, default=2025, help='random seed')
    extract_parser.add_argument('--seq_len', type=int, default=80, help='The sequence length of the input')
    extract_parser.add_argument('--token_len', type=int, default=4, help='How many time series data points we consider as a token')
    extract_parser.add_argument('--mix_embeds', action='store_true',  default=False, help='Add timestamp embedding to the inputs for the llama')
    extract_parser.add_argument('--setting', type=str, required=True, help='The full experiment name in pretrain')
    extract_parser.add_argument('--split_config_filename', type=str, required=True, help='The split config filename. The json file contains the splits of repeats and classifier data')
    extract_parser.add_argument('--enable_bp_filter', action='store_true',  default=False, help='Enable bandpass filter for preprocessing EMG')
    extract_parser.add_argument('--enable_rectify', action='store_true',  default=False, help='Enable rectification for preprocessing EMG')
    extract_parser.add_argument('--enable_lp_filter', action='store_true',  default=False, help='Enable lowpass filter for preprocessing EMG')
    extract_parser.add_argument('--cutoff_f', type=int, default=90, help='The cutoff frequency of the lowpass filter (in enabled)')
    extract_parser.add_argument('--classifier_data_dir', type=str, default='/mnt/scratchpad/xzt_ma/phase_0_exp_a/', help='The place to save data (extracted features) as h5')

    # Subparser: classifier
    classifier_parser = subparsers.add_parser("classifier")
    classifier_parser.add_argument('--config', required=True, help='Path to the config YAML file for train downstream or pure classifiers.')
    classifier_parser.add_argument('--gpu', type=int, default=0, help='gpu')
    classifier_parser.add_argument('--seed', type=int, default=2025, help='random seed')
    classifier_parser.add_argument('--train_h5_path', type=str, required=True, help='Path to the train data')
    classifier_parser.add_argument('--val_h5_path', type=str, required=True, help='Path to the val data')
    classifier_parser.add_argument('--test_h5_path', type=str, required=True, help='Path to the test data')
    classifier_parser.add_argument('--batch_size', type=int, default=64, help='Batch size for classifier training')
    classifier_parser.add_argument('--ckpt_filename', type=str, required=True, help='The filename for classifier checkpoint')
    classifier_parser.add_argument('--num_classes', type=int, required=True, help='The number of gestures (classes) including rest')
    classifier_parser.add_argument('--classifier', type=str, required=True, help='The classifier type')
    classifier_parser.add_argument('--trainer', type=str, required=True, help='The classifier trainer type')
    classifier_parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate for training classifier')
    classifier_parser.add_argument('--epochs', type=int, default=100, help='number of epochs for classifier training')
    classifier_parser.add_argument('--exp_name', type=str, required=True, help='name of current experiment name')
    classifier_parser.add_argument('--enable_balance_labels', action='store_true',  default=False, help='Balance the labels for training classifiers via downsampling')
    classifier_parser.add_argument('--downsample_target_ratio', type=float, default=1.5, help='downsampling samples number = max(other_class_count) * target_ratio')
    classifier_parser.add_argument('--enable_channel_circular_padding', action='store_true', default=False, help='Enable channel circular padding')
    classifier_parser.add_argument('--feature_dim', type=int, required=True, help='The input feature or data dimension for the classifier. ssl feature: 4096, raw data: seq_len')
    classifier_parser.add_argument('--bottleneck_dim', type=int, default=32, help='The bottleneck dimension of the mlp classifier')
    classifier_parser.add_argument('--hidden_dim', type=int, default=128, help='The hidden dimension in the classification head in mlp classifier. 0 available, means no hidden layer')

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    registry.get_task_class(args.task)(args)

if __name__ == "__main__":
    main()
