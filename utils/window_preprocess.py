import numpy as np
from scipy import signal


def window_preprocess(
    seq: np.ndarray,
    enable_bp_filter: bool = False,  # bandpass filter, fixed band 20-95Hz
    enable_rectify: bool = False,
    enable_lp_filter: bool = False,  # lowpass filter
    cutoff_f: int = 30,  # The cutoff frequency of the lowpass filter
):
    seq = seq.reshape(-1)
    fs = 200
    norm_min = -128.0
    norm_max = 127.0

    if enable_bp_filter:
        # Bandpass filter
        nyq = 0.5 * fs
        low = 20 / nyq
        high = 95 / nyq
        b, a = signal.butter(4, [low, high], btype='band')
        seq = signal.filtfilt(b, a, seq)

    # Notch filter at 50Hz
    f0 = 50
    Q = 30
    b, a = signal.iirnotch(f0, Q, fs)
    seq = signal.filtfilt(b, a, seq)

    if enable_rectify:
        seq = np.abs(seq)
        norm_min = 0.0
        norm_max = 128.0

    if enable_lp_filter:
        # Envelope extraction: lowpass filter
        low = cutoff_f / nyq
        b, a = signal.butter(4, low, btype='low')
        seq = signal.filtfilt(b, a, seq)
    
    # Min max normalization
    seq = np.clip(seq, norm_min, norm_max)
    seq = (seq - norm_min) / (norm_max - norm_min)


    return seq.reshape(-1, 1)

    
