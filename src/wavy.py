# Standard Library Imports
import sys
import math

# Third-party Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Local Imports
from utils import get_repo_path


class Wavy:
    def __init__(self, audio_file: str) -> None:
        self.audio_file = audio_file
        self.repo_path = get_repo_path()
        self.audio_dir = self.repo_path.joinpath("audio")
        self.audio_path = self.audio_dir.joinpath(audio_file)

        if self.audio_path.exists():
            self.sampling_freq, self.audio = wavfile.read(self.audio_path)
            self.audio_data_type = self.audio.dtype
            self.audio_shape = self.audio.shape
            self.max_amplitude = np.abs(self.audio).max()
            self.min_amplitude = np.abs(self.audio).min()

            self.channel_0 = self.audio[:, 0]
            if self.audio.shape[1] == 2:
                self.channel_1 = self.audio[:, 0]
        else:
            raise FileNotFoundError("Provided audio file does not exist")

    def __repr__(self) -> str:
        wavy_formal_string = str(f"<Wavy class of audio file: {self.audio_path}>")
        return wavy_formal_string

    def __str__(self) -> str:
        wavy_informal_string = str(
            "######################## AUDIO ########################\n"
            f"|Audio file:         {str(self.audio_path)}\n"
            f"|Data type:          {self.audio_data_type}\n"
            f"|Shape:              {self.audio_shape}\n"
            f"|Sampling frequency: {self.sampling_freq}\n"
            f"|Min amplitude:      {self.min_amplitude}\n"
            f"|Max amplitude:      {self.max_amplitude}\n"
            "#######################################################"
        )
        return wavy_informal_string

    def separate_silence_and_noise(
        self, channel: np.ndarray, threshold=0.0001
    ) -> tuple:
        """_summary_

        Args:
            channel (np.ndarray): _description_
            threshold (float, optional): _description_. Defaults to 0.0001.

        Returns:
            tuple: _description_
        """
        silence_parts = np.where(np.abs(channel) < threshold)[0]
        noise_parts = np.where(np.abs(channel) > threshold)[0]
        silence_samples_count = channel[silence_parts].shape[0]
        noise_samples_count = channel[noise_parts].shape[0]
        return noise_parts, noise_samples_count, silence_parts, silence_samples_count

    def calculate_time_array(
        self, sample_points: np.ndarray, sampling_freq: int
    ) -> np.ndarray:
        """_summary_

        Args:
            sample_points (np.ndarray): _description_
            sampling_freq (int): _description_

        Returns:
            np.ndarray: _description_
        """
        time_array = np.arange(0, sample_points, 1)
        time_array = time_array / sampling_freq
        time_array = time_array / 60
        return time_array

    def remove_silence(self, threshold=4) -> None:
        """_summary_

        Args:
            threshold (int, optional): _description_. Defaults to 4.
        """
        sample_points = self.audio.shape[0]
        (
            noise_parts,
            noise_samples_count,
            silence_parts,
            silence_samples_count,
        ) = self.separate_silence_and_noise(self.channel_0, threshold)

        time_array = self.calculate_time_array(sample_points, self.sampling_freq)
        silence_time_array = self.calculate_time_array(
            silence_samples_count, self.sampling_freq
        )
        noise_time_array = self.calculate_time_array(
            noise_samples_count, self.sampling_freq
        )

        original_audio_duration = math.modf(
            np.round(sample_points / self.sampling_freq / 60, 2)
        )
        og_audio_min = int(original_audio_duration[1])
        og_audio_sec = int(60 * original_audio_duration[0])

        silence_audio_duration = math.modf(
            np.round(silence_samples_count / self.sampling_freq / 60, 2)
        )
        silence_audio_min = int(silence_audio_duration[1])
        silence_audio_sec = int(60 * silence_audio_duration[0])
        noise_audio_duration = math.modf(
            np.round(noise_samples_count / self.sampling_freq / 60, 2)
        )
        noise_audio_min = int(noise_audio_duration[1])
        noise_audio_sec = int(60 * noise_audio_duration[0])

        self.time_arrays = [time_array, silence_time_array, noise_time_array]
        self.audio_arrays = [
            self.audio,
            self.audio[silence_parts, :],
            self.audio[noise_parts, :],
        ]

        audio_no_silence_file = self.audio_file.removesuffix(".wav") + "_no_silence.wav"
        audio_no_silence_file_path = self.audio_dir.joinpath(audio_no_silence_file)

        print("################## Removing silence ###################")
        print(f"|Audio without silence file: {audio_no_silence_file_path}")
        print(f"|")
        print(f"|Original samples: {time_array.shape[0]}")
        print(f"|Original audio duration: {og_audio_min}min {og_audio_sec}s")
        print(f"|")
        print(f"|Silence samples: {silence_time_array.shape[0]}")
        print(f"|Silence audio duration: {silence_audio_min}min {silence_audio_sec}s")
        print(f"|")
        print(f"|Noise samples: {noise_time_array.shape[0]}")
        print(f"|Noise audio duration: {noise_audio_min}min {noise_audio_sec}s")
        print("#######################################################")

        wavfile.write(
            audio_no_silence_file_path,
            self.sampling_freq,
            self.audio[noise_parts].astype(self.audio_data_type),
        )

    def plot_amplitude(self):
        """Plot orginal audio, silence and noise parts amplitudes"""
        fig, axs = plt.subplots(3, 1)
        fig.suptitle("Audio amplitude", fontsize=16)
        fig.set_figheight(10)
        fig.set_figwidth(16)
        axs[0].plot(self.time_arrays[0], self.audio_arrays[0], color="r")
        axs[0].set_title("Original audio")
        axs[0].set_ylabel("Amplitude")
        axs[1].plot(self.time_arrays[1], self.audio_arrays[1], color="g")
        axs[1].set_title("silence parts")
        axs[1].set_ylabel("Amplitude")
        axs[2].plot(self.time_arrays[2], self.audio_arrays[2], color="b")
        axs[2].set_title("noise parts")
        axs[2].set_xlabel("Duration (min)")
        axs[2].set_ylabel("Amplitude")
        plt.show()


def main():
    test_wavy = Wavy("test.wav")
    print(repr(test_wavy))
    print(test_wavy)
    test_wavy.remove_silence_below_threshold()
    test_wavy.plot_amplitude()
    return 0


if __name__ == "__main__":
    sys.exit(main())
