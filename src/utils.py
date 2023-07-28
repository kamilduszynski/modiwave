# Standard Library Imports
import os
from pathlib import Path, PosixPath


def get_repo_path() -> PosixPath:
    """Returns absolute path of this repository as a string"""
    current_path = Path(os.path.abspath(".."))
    this_file_path = current_path / Path(__file__)
    return this_file_path.parent.parent


def get_sample_index_by_time(start_time, end_time, sampling_rate):
    start_sample = int(start_time * sampling_rate)
    end_sample = int(end_time * sampling_rate)
    return start_sample, end_sample


def cut_audio_segment(start_sample, end_sample, audio_array):
    return audio_array[start_sample : end_sample + 1]
