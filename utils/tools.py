import yaml
import torch
import numpy as np
import math
from collections import defaultdict


class DotDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError(f"No such attribute: {name}")


def load_and_merge_config(args):
    """
    Merge args and configs from yaml files. If a conflict occurs, take the values from args.
    """
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    for key, value in config_dict.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    args_dict = vars(args)
    return DotDict(args_dict)


def find_matching_dict(gestures_list, gesture_dicts):
    for ex, d in gesture_dicts.items():
        if all(g in d for g in gestures_list):
            return ex, d
    raise ValueError("Please check if gestures_list is valid or contains spelling mistakes.")


def track_gestures(gestures_list, gesture_dicts_dict):
    tracks = {}
    for gesture in gestures_list:
        for ex, gesture_dict in gesture_dicts_dict.items():
            if gesture in gesture_dict:
                if ex not in tracks:
                    tracks[ex] = (gesture_dict, [])
                tracks[ex][1].append(gesture)
    
    return tracks


class EarlyStopping:
    def __init__(self, args, verbose=False, delta=0):
        self.patience = args.patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            self.counter = 0


    def save_checkpoint(self, val_loss, model, path):
        param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
        }
        state_dict = model.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        torch.save(state_dict, path + '/' + f'checkpoint.pth')


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** (epoch - 1))}    
    elif args.lradj == 'type2':
        lr_adjust = {epoch: args.learning_rate * (0.6 ** epoch)}  
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('next learning rate is {}'.format(lr))


class ClassifierEarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_acc = 0
        self.best_model = None

    def step(self, model, acc):
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_model = model.state_dict()
            self.counter = 0
            return False  # not early stop
        else:
            self.counter += 1
            return self.counter >= self.patience


def downsample_dataset(dataset, downsample_label=0, target_ratio=1.5, flag='training'):
    """
    Downsample the specified label in the dataset and return a new list of indices.

    Args:
    - dataset: a torch.utils.data.Dataset instance, where dataset[i][1] returns the label.
    - downsample_label: the class label to downsample (e.g., 0).
    - target_ratio: the target number of samples for the downsample_label is 
                    max(other_class_count) * target_ratio.

    Returns:
    - final_indices: A list of indices to retain in the dataset after downsampling.
    - original_label_counts: A dict showing original number of samples per label.
    - final_label_counts: A dict showing final (after downsampling) number of samples per label.
    """
    print(f"Downsampling on label {downsample_label} for the {flag} set, could take a while ...\n")
    label_to_indices = defaultdict(list)
    
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = label.item()
        label_to_indices[label].append(idx)

    # Original label count
    original_label_counts = {k: len(v) for k, v in label_to_indices.items()}
    
    # Compute the maximum number of samples among all other classes
    other_class_counts = [len(v) for k, v in label_to_indices.items() if k != downsample_label]
    max_other_class = max(other_class_counts)
    target_count = int(max_other_class * target_ratio)

    # Perform downsampling if needed
    downsampled_indices = (
        np.random.choice(label_to_indices[downsample_label], size=target_count, replace=False).tolist()
        if len(label_to_indices[downsample_label]) > target_count
        else label_to_indices[downsample_label]
    )

    # Combine downsampled and all other class indices
    final_indices = downsampled_indices
    for k, v in label_to_indices.items():
        if k != downsample_label:
            final_indices.extend(v)

    np.random.shuffle(final_indices)

    # Compute label count after downsampling
    final_label_counts = defaultdict(int)
    for idx in final_indices:
        _, label = dataset[idx]
        label = label.item()
        final_label_counts[label] += 1

    return final_indices, original_label_counts, dict(final_label_counts)
