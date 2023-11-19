# Standard Library Imports
import os
from pathlib import Path, PosixPath

# Third-party Imports
import numpy as np
from tqdm import tqdm


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
    segment = np.arange(start=start_sample, stop=end_sample)
    return np.delete(audio_array, segment)


def calculate_time_array(samples_count: int, sampling_rate: int) -> np.ndarray:
    return np.linspace(0, samples_count / sampling_rate, num=samples_count)


def rolling_window(arr, window):
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def find_subarrays(arr_a, arr_b):
    temp = rolling_window(arr_a, len(arr_b))

    # result = np.where(np.all(temp == b, axis=1))
    result = []

    for i in tqdm(range(temp.shape[0])):
        if (temp[i, :] == arr_b).all():
            result.append(i)

    # for i, row in tqdm(enumerate(temp)):
    #     if (row == b).all():
    #         result.append(i)

    return np.array(result)  # [0] if result else None
