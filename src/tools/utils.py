# Standard Library Imports
import os
from pathlib import Path, PosixPath

# Third-party Imports
import numpy as np


def get_repo_path() -> PosixPath:
    """Returns absolute path of this repository as a string"""
    current_path = Path(os.path.abspath(".."))
    this_file_path = current_path / Path(__file__)
    return this_file_path.parent.parent.parent


def get_sample_index_by_time(
    start_time: float, end_time: float, sampling_rate: int
) -> tuple:
    start_sample = int(start_time * sampling_rate)
    end_sample = int(end_time * sampling_rate)
    return start_sample, end_sample


def cut_audio_segment(
    start_sample: float, end_sample: float, audio_array: np.ndarray
) -> np.ndarray:
    return audio_array[start_sample : end_sample + 1]


def calculate_time_array(samples_count: int, sampling_rate: int) -> np.ndarray:
    return np.linspace(0, samples_count / sampling_rate, num=samples_count)


def convert_to_decibel(sample: float) -> float:
    ref = 1
    if sample != 0:
        return 20 * np.log10(abs(sample) / ref)
    else:
        return -60


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def find_subarrays(a, b):
    temp = rolling_window(a, len(b))
    result = np.where(np.all(temp == b, axis=1))
    return result[0] if result else None
