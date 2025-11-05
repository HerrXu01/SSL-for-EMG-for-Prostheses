import os
import random
import json
import numpy as np
from scipy.io import loadmat
from utils.tools import find_matching_dict, track_gestures
from utils.label_map import (
    NINAPRO_DB5_EX_A_GESTURE_TO_LABEL,
    NINAPRO_DB5_EX_B_GESTURE_TO_LABEL,
    NINAPRO_DB5_EX_C_GESTURE_TO_LABEL,
    NINAPRO_DB5_ALL_GESTURE_TO_LABEL,
    EXP_TO_GESTURES_LIST
)
from data_provider.datasets import EMGPretrainDataset
from torch.utils.data import DataLoader, ConcatDataset


GESTURE_DICTS_DICT = {
    1: NINAPRO_DB5_EX_A_GESTURE_TO_LABEL,
    2: NINAPRO_DB5_EX_B_GESTURE_TO_LABEL,
    3: NINAPRO_DB5_EX_C_GESTURE_TO_LABEL
}


def get_gestures_split(gestures_list, matching_dict, all_emg_data, restimulus):
    if restimulus.ndim == 2:
        restimulus = restimulus.flatten()

    gestures_split = {}
    for gesture in gestures_list:
        target = matching_dict[gesture]
        if target == 1:
            latter_start_idx = np.where(restimulus == 2)[-1][0]
            target_start_idx = np.where(restimulus == target)[-1][0]
            target_end_idx = np.where(restimulus == target)[-1][-1]
            start_cut = int(target_start_idx/2)  # To cut out the noisy part at the beginning
            end_cut = int((target_end_idx + latter_start_idx)/2)
            gestures_split[gesture] = (all_emg_data[start_cut:end_cut, :], restimulus[start_cut:end_cut])
        elif target == restimulus.max():
            former = restimulus.max() - 1
            former_end_idx = np.where(restimulus == former)[-1][-1]
            target_start_idx = np.where(restimulus == target)[-1][0]
            start_cut = int((former_end_idx + target_start_idx)/2)
            gestures_split[gesture] = (all_emg_data[start_cut:, :], restimulus[start_cut:])
        else:
            former = target - 1
            latter = target + 1
            former_end_idx = np.where(restimulus == former)[-1][-1]
            latter_start_idx = np.where(restimulus == latter)[-1][0]
            target_start_idx = np.where(restimulus == target)[-1][0]
            target_end_idx = np.where(restimulus == target)[-1][-1]
            start_cut = int((former_end_idx + target_start_idx)/2)
            end_cut = int((target_end_idx + latter_start_idx)/2)
            gestures_split[gesture] = (all_emg_data[start_cut:end_cut, :], restimulus[start_cut:end_cut])
    
    return gestures_split


def get_repeats_split(gestures_split):
    gestures_repeats_split = {}
    for gesture, (gesture_emg, gesture_restimulus) in gestures_split.items():
        unique = np.unique(gesture_restimulus)
        assert unique[0] == 0 and len(unique) == 2, f"fixed gesture restimulus contains more than 2 different numbers {unique}, please check the dataset"
        d = np.diff(gesture_restimulus)
        starts = np.where(d > 0)[0] + 1
        ends = np.where(d < 0)[0] + 1

        if gesture_restimulus[0] > 0:
            starts = np.r_[0, starts]
        if gesture_restimulus[-1] > 0:
            ends = np.r_[ends, len(gesture_restimulus)]

        N = len(starts)
        if N != 6:
            raise ValueError(f"Current gesture {gesture} doesn't contain 6 repeats, but {N} repeats. Please check the dataset.")
        
        mids = ((ends[:-1] + starts[1:]) // 2).tolist()
        repeats = np.split(gesture_emg, mids)
        repeats_restimulus = np.split(gesture_restimulus, mids)
        gestures_repeats_split[gesture] = (repeats, repeats_restimulus)

    return gestures_repeats_split


def get_dataset_ninapro_db5(args):
    if args.subjects_list == 'all':
        subjects_list = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
    else:
        subjects_list = args.subjects_list

    if isinstance(args.gestures_list, str):
        gestures_list = EXP_TO_GESTURES_LIST[args.gestures_list]
    else:
        gestures_list = args.gestures_list

    # ex, matching_dict = find_matching_dict(gestures_list, GESTURE_DICTS_DICT)
    tracks = track_gestures(gestures_list, GESTURE_DICTS_DICT)

    initial_repeats = [0, 1, 2, 3, 4, 5]
    repeats_split_dict = {}
    ssl_train_datasets = []
    ssl_val_datasets = []

    for subject_id in subjects_list:
        repeats_split_dict[subject_id] = {}
        subject_id_upper = subject_id.upper()
        sbj_gestures_repeats_split_lists = []
        for ex, (matching_dict, sub_gestures_list) in tracks.items():
            data_filename = f'{subject_id_upper}_E{ex}_A1.mat'
            data_file_path = os.path.join(args.dataset_root_path, subject_id, data_filename)
        
            try:
                mat_data_dict = loadmat(data_file_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load file: {data_file_path}. "
                    f"Please check if dataset_root_path, subjects_list, and gestures_list are correct.\n"
                    f"Original error: {e}"
                )

            all_emg_data = mat_data_dict["emg"][:, :8]
            gestures_split = get_gestures_split(
                sub_gestures_list,
                matching_dict,
                all_emg_data,
                mat_data_dict["restimulus"]
            )
            gestures_repeats_split = get_repeats_split(gestures_split)
            sbj_gestures_repeats_split_lists.append(gestures_repeats_split)
        # Here sbj_gestures_repeats_split_dict is a dict in which the keys are pre-set gestures,
        # the corresponding value is a tuple of two lists
        # The first list (len 6) contains the repeat segments (6 2-d arrays) of that gesture in order.
        # The second list (also len 6) contains the corresonding restimuslus (labels).
        sbj_gestures_repeats_split_dict = {k: v for d in sbj_gestures_repeats_split_lists for k, v in d.items()}

        for gesture, (repeats_list, restimulus_list) in sbj_gestures_repeats_split_dict.items():
            current_sample = random.sample(initial_repeats, 6)
            repeats_split_dict[subject_id][gesture] = {}
            repeats_split_dict[subject_id][gesture]["ssl_train_repeats_id"] = []
            for repeat_id in current_sample[:3]:
                repeats_split_dict[subject_id][gesture]["ssl_train_repeats_id"].append(repeat_id)
                single_repeat_dataset = EMGPretrainDataset(
                    repeats_list[repeat_id],
                    seq_len=args.seq_len,
                    token_len=args.token_len,
                    enable_bp_filter=args.enable_bp_filter,
                    enable_rectify=args.enable_rectify,
                    enable_lp_filter=args.enable_lp_filter,
                    cutoff_f=args.cutoff_f
                )
                ssl_train_datasets.append(single_repeat_dataset)

            repeats_split_dict[subject_id][gesture]["ssl_val_repeats_id"] = [current_sample[3]]
            ssl_val_datasets.append(EMGPretrainDataset(
                repeats_list[current_sample[3]],
                seq_len=args.seq_len,
                token_len=args.token_len,
                enable_bp_filter=args.enable_bp_filter,
                enable_rectify=args.enable_rectify,
                enable_lp_filter=args.enable_lp_filter,
                cutoff_f=args.cutoff_f
            ))

            repeats_split_dict[subject_id][gesture]["classifier_train_repeats_id"] = [current_sample[4]]
            # Here we have to store the data in the form of list. JSON cannot store numpy array. Remember to recover the data to numpy array when loading for classifier
            repeats_split_dict[subject_id][gesture]["classifier_train_data"] = repeats_list[current_sample[4]].tolist()
            repeats_split_dict[subject_id][gesture]["classifier_train_restimulus"] = restimulus_list[current_sample[4]].tolist()
            repeats_split_dict[subject_id][gesture]["classifier_valtest_repeats_id"] = [current_sample[5]]
            repeats_split_dict[subject_id][gesture]["classifier_valtest_data"] = repeats_list[current_sample[5]].tolist()
            repeats_split_dict[subject_id][gesture]["classifier_valtest_restimulus"] = restimulus_list[current_sample[5]].tolist()
    
    json_save_path = os.path.join(args.split_path, args.split_filename)
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(repeats_split_dict, f, indent=4)

    ssl_train_datasets_combined = ConcatDataset(ssl_train_datasets)
    ssl_val_datasets_combined = ConcatDataset(ssl_val_datasets)

    return ssl_train_datasets_combined, ssl_val_datasets_combined


def data_provider(args):
    ssl_train_set, ssl_val_set = get_dataset_ninapro_db5(args)

    print(f"SSL Train Dataset Length: {len(ssl_train_set)}")
    print(f"SSL Val Dataset Length: {len(ssl_val_set)}")

    ssl_train_loader = DataLoader(
        ssl_train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False
    )

    ssl_val_loader = DataLoader(
        ssl_val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )

    return ssl_train_loader, ssl_val_loader

    