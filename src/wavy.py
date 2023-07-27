# Standard Library Imports
import re
import sys
import math

# Third-party Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from .utils import get_repo_path


class Wavy:
    def __init__(self, audio_file: str) -> None:
        self.audio_file = audio_file
        self.repo_path = get_repo_path()
        self.audio_dir = self.repo_path.joinpath("audio")
        self.audio_path = self.audio_dir.joinpath(audio_file)

        if self.audio_path.exists():
            self.sampling_rate, self.audio = wavfile.read(self.audio_path)
            self.audio_data_type = self.audio.dtype
            self.audio_shape = self.audio.shape
            self.audio_samples = self.audio.shape[0]
            self.audio_duration = self.audio_samples / self.sampling_rate
            self.l_channel = self.audio[:, 0]
            self.max_amplitude = np.abs(self.l_channel).max()
            self.min_amplitude = np.abs(self.l_channel).min()

            if self.audio.shape[1] == 2:
                self.r_channel = self.audio[:, 1]
            else:
                self.r_channel = None
        else:
            raise FileNotFoundError("Provided audio file does not exist")

    def __repr__(self) -> str:
        wavy_formal_string = str(
            f"<{self.__class__.__name__} class of audio file: {self.audio_path}>"
        )
        return wavy_formal_string

    def __str__(self) -> str:
        wavy_informal_string = str(
            "######################## AUDIO ########################\n"
            f"|Audio file:         {str(self.audio_path)}\n"
            f"|Data type:          {self.audio_data_type}\n"
            f"|Shape:              {self.audio_shape}\n"
            f"|Sampling rate:      {self.sampling_rate}\n"
            f"|Duration:           {self.audio_duration}s\n"
            f"|Min amplitude:      {self.min_amplitude}\n"
            f"|Max amplitude:      {self.max_amplitude}\n"
            "#######################################################"
        )
        return wavy_informal_string

    def __calculate_time_array(self, samples_count: int) -> np.ndarray:
        time_array = np.linspace(
            0, samples_count / self.sampling_rate, num=samples_count
        )
        return time_array

    def __remove_silence_plot(self, time_arrays, audio_arrays):
        fig, axs = plt.subplots(3, 1)
        fig.suptitle("Audio amplitude", fontsize=16)
        fig.set_figheight(10)
        fig.set_figwidth(16)
        axs[0].plot(time_arrays[0], audio_arrays[0], color="r")
        axs[0].set_title("Original audio")
        axs[0].set_ylabel("Amplitude")
        axs[1].plot(time_arrays[1], audio_arrays[1], color="g")
        axs[1].set_title("Silence parts")
        axs[1].set_ylabel("Amplitude")
        axs[2].plot(time_arrays[2], audio_arrays[2], color="b")
        axs[2].set_title("Noise parts")
        axs[2].set_xlabel("Duration (s)")
        axs[2].set_ylabel("Amplitude")
        plt.show()

    def remove_silence(self, threshold=4, plot=False) -> None:
        audio_samples_count = self.audio.shape[0]
        noise_samples_indexes = np.where(np.abs(self.l_channel) > threshold)[0]
        noise_samples_count = self.l_channel[noise_samples_indexes].shape[0]
        silence_samples_indexes = np.where(np.abs(self.l_channel) < threshold)[0]
        silence_samples_count = self.l_channel[silence_samples_indexes].shape[0]

        original_audio_duration = math.modf(
            np.round(audio_samples_count / self.sampling_rate / 60, 2)
        )
        og_audio_min = int(original_audio_duration[1])
        og_audio_sec = int(60 * original_audio_duration[0])

        silence_audio_duration = math.modf(
            np.round(silence_samples_count / self.sampling_rate / 60, 2)
        )
        silence_audio_min = int(silence_audio_duration[1])
        silence_audio_sec = int(60 * silence_audio_duration[0])
        noise_audio_duration = math.modf(
            np.round(noise_samples_count / self.sampling_rate / 60, 2)
        )
        noise_audio_min = int(noise_audio_duration[1])
        noise_audio_sec = int(60 * noise_audio_duration[0])

        regx = re.compile(".wav\\b")
        audio_no_silence_file = regx.sub("_no_silence.wav", self.audio_file)
        audio_no_silence_file_path = self.audio_dir.joinpath(audio_no_silence_file)

        print("################## Removing silence ###################")
        print(f"|Audio without silence file: {audio_no_silence_file_path}")
        print(f"|")
        print(f"|Original samples: {audio_samples_count}")
        print(f"|Original audio duration: {og_audio_min}min {og_audio_sec}s")
        print(f"|")
        print(f"|Silence samples: {silence_samples_count}")
        print(f"|Silence audio duration: {silence_audio_min}min {silence_audio_sec}s")
        print(f"|")
        print(f"|Noise samples: {noise_samples_count}")
        print(f"|Noise audio duration: {noise_audio_min}min {noise_audio_sec}s")
        print("#######################################################")

        wavfile.write(
            audio_no_silence_file_path,
            self.sampling_rate,
            self.audio[noise_samples_indexes].astype(self.audio_data_type),
        )

        if plot:
            time_array = self.__calculate_time_array(audio_samples_count)
            silence_time_array = self.__calculate_time_array(silence_samples_count)
            noise_time_array = self.__calculate_time_array(noise_samples_count)
            time_arrays = [time_array, silence_time_array, noise_time_array]
            audio_arrays = [
                self.audio,
                self.audio[silence_samples_indexes, :],
                self.audio[noise_samples_indexes, :],
            ]
            self.__remove_silence_plot(time_arrays, audio_arrays)

    def plot_audio_amplitude(self):
        """Plot orginal audio signal amplitude"""
        fig, axs = plt.subplots(3, 1)
        fig.suptitle("Audio amplitude", fontsize=16)
        fig.set_figheight(10)
        fig.set_figwidth(16)
        axs[0].plot(
            self.__calculate_time_array(self.audio_samples), self.l_channel, color="r"
        )
        axs[0].set_title("Left channel")
        axs[0].set_ylabel("Amplitude")
        if self.r_channel is not None:
            axs[1].plot(
                self.__calculate_time_array(self.audio_samples),
                self.r_channel,
                color="b",
            )
            axs[1].set_title("Right channel")
            axs[1].set_ylabel("Amplitude")
            axs[2].plot(
                self.__calculate_time_array(self.audio_samples),
                self.l_channel - self.r_channel,
                color="g",
            )
            axs[2].set_title("Channels difference")
            axs[2].set_ylabel("Amplitude")
            axs[2].set_xlabel("Duration (s)")
        plt.show()

    def plot_audio_frequency_spectrum(self):
        """Plot orginal audio frequency spectrum"""
        plt.figure(figsize=(16, 10))
        plt.suptitle("Audio frequency spectrum", fontsize=16)
        plt.specgram(self.l_channel, Fs=self.sampling_rate, vmin=-20, vmax=40)
        plt.xlabel("Duration (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar()
        plt.show()


def main():
    test_wavy = Wavy("test.wav")
    print(repr(test_wavy))
    print(test_wavy)
    test_wavy.remove_silence(plot=True)
    test_wavy.plot_audio_amplitude()
    test_wavy.plot_audio_frequency_spectrum()
    return 0


if __name__ == "__main__":
    sys.exit(main())
