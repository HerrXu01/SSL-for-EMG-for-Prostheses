import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.window_preprocess import window_preprocess


class EMGPretrainDataset(Dataset):
    def __init__(self, 
    emg_data: np.ndarray,
    seq_len: int,
    token_len: int,
    enable_bp_filter: bool = False,
    enable_rectify: bool = False,
    enable_lp_filter: bool = False,
    cutoff_f: int = 30,
    ):
        self.emg_data = emg_data
        self.window_len = seq_len + token_len
        self.num_time_steps = emg_data.shape[0]
        self.num_channels = emg_data.shape[1]
        self.enable_bp_filter = enable_bp_filter
        self.enable_rectify = enable_rectify
        self.enable_lp_filter = enable_lp_filter
        self.cutoff_f = cutoff_f
        self.seq_len = seq_len

        self.windows_per_channel = self.num_time_steps - self.window_len + 1
        if self.windows_per_channel <= 0:
            raise ValueError(f"The window length (seq_len + token_len) is too large, should be smaller than the time steps in the repeats: {self.num_time_steps}")

        self.total_windows = self.windows_per_channel * self.num_channels

    def __getitem__(self, index):
        channel = index // self.windows_per_channel
        offset = index % self.windows_per_channel
        window = self.emg_data[offset:offset+self.window_len, channel:channel+1]
        window = window_preprocess(
            window,
            enable_bp_filter=self.enable_bp_filter,
            enable_rectify=self.enable_rectify,
            enable_lp_filter=self.enable_lp_filter,
            cutoff_f=self.cutoff_f
        )

        seq_x = window[:self.seq_len]
        seq_y = window[-self.seq_len:]

        return seq_x, seq_y

    def __len__(self):
        return self.total_windows


class EMGClassifierDataset(Dataset):
    def __init__(self, h5_path):
        super().__init__()
        self.h5_file = h5py.File(h5_path, "r")
        self.inputs = self.h5_file["inputs"]
        self.labels = self.h5_file["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.inputs[idx]  # (8, num_tokens, 4096)
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

