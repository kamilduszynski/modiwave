# Standard Library Imports
import os
import re
import csv
import sys
import json
import math
import wave

# Third-party Imports
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from vosk import Model, KaldiRecognizer
from scipy.io import wavfile

# Local Imports
import src.tools.utils as utils
from src.tools.word import Word


class Wavy:
    def __init__(self, audio_file: str) -> None:
        self.audio_file = audio_file
        self.regx = re.compile(".wav\\b")
        self.repo_path = utils.get_repo_path()
        self.audio_dir = self.repo_path.joinpath("audio")
        if not self.audio_dir.exists():
            os.mkdir(self.audio_dir)
        self.transcripts_dir = self.repo_path.joinpath("transcripts")
        if not self.transcripts_dir.exists():
            os.mkdir(self.transcripts_dir)
        transcript_file = self.regx.sub("_transcript.csv", self.audio_file)
        self.transcript_file_path = self.transcripts_dir.joinpath(transcript_file)
        self.audio_path = self.audio_dir.joinpath(audio_file)

        if self.audio_path.exists():
            self.sampling_rate, self.audio = wavfile.read(self.audio_path)
            self.audio_data_type = self.audio.dtype
            self.audio_shape = self.audio.shape
            self.audio_samples = self.audio.shape[0]
            self.audio_duration = self.audio_samples / self.sampling_rate
            db_audio = librosa.amplitude_to_db(self.audio / (2.0**15))

            if len(self.audio_shape) == 2:
                self.l_channel = self.audio[:, 0]
                self.r_channel = self.audio[:, 1]
                self.db_audio_l = db_audio[:, 0]
                self.db_audio_r = db_audio[:, 1]
            else:
                self.l_channel = self.audio
                self.r_channel = None
                self.db_audio_l = db_audio
                self.db_audio_r = None
            self.max_amplitude = np.abs(self.l_channel).max()
            self.min_amplitude = np.abs(self.l_channel).min()
        else:
            raise FileNotFoundError("Provided audio file does not exist")

    def __repr__(self) -> str:
        wavy_formal_string = str(
            f"<class '{self.__class__.__name__}' from audio file: {self.audio_path}>"
        )
        return wavy_formal_string

    def __str__(self) -> str:
        wavy_informal_string = str(
            f"{'#'*20} AUDIO {'#'*20}\n"
            f"|Audio file:         {str(self.audio_path)}\n"
            f"|Data type:          {self.audio_data_type}\n"
            f"|Shape:              {self.audio_shape}\n"
            f"|Sampling rate:      {self.sampling_rate}\n"
            f"|Duration:           {self.audio_duration}s\n"
            f"|Min amplitude:      {self.min_amplitude}\n"
            f"|Max amplitude:      {self.max_amplitude}\n"
            f"{'#'*60}"
        )
        return wavy_informal_string

    def _save_audio_mono(self) -> None:
        audio_mono_file = self.regx.sub("_mono.wav", self.audio_file)
        self.audio_mono_file_path = self.audio_dir.joinpath(audio_mono_file)
        wavfile.write(
            self.audio_mono_file_path,
            self.sampling_rate,
            self.l_channel.astype(self.audio_data_type),
        )

    def _remove_silence_plot(self, db_scale: bool) -> None:
        time_array = utils.calculate_time_array(self.audio_samples, self.sampling_rate)
        silence_time_array = utils.calculate_time_array(
            self.silence_samples_count, self.sampling_rate
        )
        no_silence_time_array = utils.calculate_time_array(
            self.no_silence_samples_count, self.sampling_rate
        )
        time_arrays = (time_array, no_silence_time_array, silence_time_array)
        if db_scale:
            audio_arrays = (
                self.db_audio_l,
                self.db_audio_l[self.no_silence_samples_indexes],
                self.db_audio_l[self.silence_samples_indexes],
            )
        else:
            audio_arrays = (
                self.l_channel,
                self.l_channel[self.no_silence_samples_indexes],
                self.l_channel[self.silence_samples_indexes],
            )
        fig, axs = plt.subplots(3, 1)
        fig.suptitle("Audio amplitude", fontsize=16)
        fig.set_figheight(10)
        fig.set_figwidth(16)
        axs[0].plot(time_arrays[0], audio_arrays[0], color="r", label="original audio")
        axs[0].plot(
            time_arrays[0][self.silence_samples_indexes],
            audio_arrays[2],
            color="c",
            label="silence",
        )
        axs[0].legend()
        axs[0].set_title("Original audio")
        axs[0].set_ylabel("Amplitude")
        axs[1].plot(time_arrays[1], audio_arrays[1], color="g")
        axs[1].set_title("No silence parts")
        axs[1].set_ylabel("Amplitude")
        axs[2].plot(time_arrays[2], audio_arrays[2], color="b")
        axs[2].set_title("Silence parts")
        axs[2].set_ylabel("Amplitude")
        axs[2].set_xlabel("Duration (s)")
        plt.show()

    def _remove_silence_print(self, audio_no_silence_file_path: str) -> None:
        og_d = math.modf(np.round(self.audio_samples / self.sampling_rate / 60, 2))
        s_d = math.modf(
            np.round(self.silence_samples_count / self.sampling_rate / 60, 2)
        )
        n_s_d = math.modf(
            np.round(self.no_silence_samples_count / self.sampling_rate / 60, 2)
        )
        print(f"{'#'*20} Removing silence {'#'*20}")
        print(f"|Audio without silence file: {audio_no_silence_file_path}")
        print(f"|")
        print(f"|Original samples: {self.audio_samples}")
        print(f"|Original duration: {int(og_d[1])}min {int(60 * og_d[0])}s")
        print(f"|")
        print(f"|No silence samples: {self.no_silence_samples_count}")
        print(f"|No silence duration: {int(n_s_d[1])}min {int(60 * n_s_d[0])}s")
        print(f"|")
        print(f"|Silence samples: {self.silence_samples_count}")
        print(f"|Silence duration: {int(s_d[1])}min {int(60 * s_d[0])}s")
        print(f"{'#'*60}")

    def remove_silence_by_signal_aplitude_threshold(
        self, threshold=4, plot=False, verbose=False
    ) -> str:
        """Remove silence with specified audio signal amplitude threshold

        Args:
            threshold (int, optional): Maximum audio signal amplitude threshold. Defaults to 4.
            plot (bool, optional): Show plot. Defaults to False.
            verbose (bool, optional): Show message. Defaults to False.

        Returns:
            str: Audio without silence filepath
        """
        self.silence_samples_indexes = np.where(np.abs(self.l_channel) < threshold)[0]
        self.silence_samples_count = self.l_channel[self.silence_samples_indexes].shape[
            0
        ]
        self.no_silence_samples_indexes = np.where(np.abs(self.l_channel) >= threshold)[
            0
        ]
        self.no_silence_samples_count = self.l_channel[
            self.no_silence_samples_indexes
        ].shape[0]

        audio_no_silence_file = self.regx.sub("_no_silence.wav", self.audio_file)
        audio_no_silence_file_path = self.audio_dir.joinpath(audio_no_silence_file)

        wavfile.write(
            audio_no_silence_file_path,
            self.sampling_rate,
            self.audio[self.no_silence_samples_indexes].astype(self.audio_data_type),
        )

        if verbose:
            self._remove_silence_print(audio_no_silence_file_path)

        if plot:
            self._remove_silence_plot(db_scale=False)
        return str(audio_no_silence_file_path)

    def remove_silence_by_db_threshold(
        self, db_threshold=30, min_duration=0.1, plot=False, verbose=False
    ) -> str:
        """Remove silence with specified minimal duration and threshold in decibels

        Args:
            db_threshold (int, optional): Maximum silence threshold in decibels. Defaults to 30.
            min_duration (float, optional): Minimum silence duration in seconds. Defaults to 0.1.
            plot (bool, optional): Show plot. Defaults to False.
            verbose (bool, optional): Show message. Defaults to Flase.

        Returns:
            str: Audio without silence filepath
        """
        min_samples = int(min_duration * self.sampling_rate)
        no_silence_samples_indexes = np.where(self.db_audio_l < db_threshold)[0]

        diff = np.diff(no_silence_samples_indexes)
        diff = np.append(diff, 1)
        diff = utils.find_subarrays(diff, np.full(min_samples, 1))

        si = np.array([])
        for i in tqdm(range(int(len(no_silence_samples_indexes) / min_samples))):
            if i * min_samples + min_samples > len(diff):
                continue
            if (
                diff[i * min_samples + min_samples] - diff[i * min_samples]
                == min_samples
            ):
                si = np.concatenate(
                    (si, diff[i * min_samples : i * min_samples + min_samples])
                )
        # rolled_diff = utils.rolling_window(diff, min_samples)
        # print(rolled_diff[0])
        # rolled_diff = np.apply_along_axis(lambda x: x[min_samples-1]-x[0], 1, rolled_diff)
        # indicies = np.arange(min_samples-1, len(no_silence_samples_indexes), min_samples-1)

        # l = int(len(no_silence_samples_indexes) / min_samples)
        # result = np.split(diff[:-1], l-1)

        self.no_silence_samples_indexes = no_silence_samples_indexes[si.astype(int)]
        self.no_silence_samples_count = self.l_channel[
            self.no_silence_samples_indexes
        ].shape[0]

        # non_silent_intervals = librosa.effects.split(
        #     self.l_channel,
        #     top_db=db_threshold,
        #     frame_length=min_samples,
        #     hop_length=min_samples,
        # )
        # print(non_silent_intervals)

        self.silence_samples_indexes = np.setdiff1d(
            np.argwhere(self.l_channel),
            np.argwhere(self.l_channel[self.no_silence_samples_indexes]),
        )
        self.silence_samples_count = self.l_channel[self.silence_samples_indexes].shape[
            0
        ]

        audio_no_silence_file = self.regx.sub("_no_silence.wav", self.audio_file)
        audio_no_silence_file_path = self.audio_dir.joinpath(audio_no_silence_file)

        wavfile.write(
            audio_no_silence_file_path,
            self.sampling_rate,
            self.audio[self.no_silence_samples_indexes].astype(self.audio_data_type),
        )

        if verbose:
            self._remove_silence_print(audio_no_silence_file_path)

        if plot:
            self._remove_silence_plot(db_scale=True)
        return str(audio_no_silence_file_path)

    def plot_audio_amplitude(self, db_scale=False) -> None:
        """Plot orginal audio signal amplitude"""
        if self.r_channel is not None:
            fig, axs = plt.subplots(3, 1)
            fig.suptitle("Audio amplitude", fontsize=16)
            fig.set_figheight(10)
            fig.set_figwidth(16)
            x = utils.calculate_time_array(self.audio_samples, self.sampling_rate)
            if db_scale:
                y = self.db_audio_l
            else:
                y = self.l_channel
            axs[0].plot(x, y, color="r")
            axs[0].set_title("Left channel")
            axs[0].set_ylabel("Amplitude")
            if db_scale:
                y = self.db_audio_r
            else:
                y = self.r_channel
            axs[1].plot(x, y, color="g")
            axs[1].set_title("Right channel")
            axs[1].set_ylabel("Amplitude")
            if db_scale:
                y = self.db_audio_r - self.db_audio_l
            else:
                y = self.r_channel - self.l_channel
            axs[2].plot(x, y, color="b")
            axs[2].set_title("Difference between channels")
            axs[2].set_ylabel("Amplitude")
            axs[2].set_xlabel("Duration (s)")
        else:
            plt.figure(figsize=(16, 10))
            plt.title("Audio amplitude", fontsize=16)
            x = utils.calculate_time_array(self.audio_samples, self.sampling_rate)
            if db_scale:
                y = self.db_audio_l
            else:
                y = self.l_channel
            plt.plot(x, y, color="r")
            plt.ylabel("Amplitude")
            plt.xlabel("Duration (s)")
        plt.show()

    def plot_audio_frequency_spectrum(self) -> None:
        """Plot orginal audio frequency spectrum"""
        plt.figure(figsize=(16, 10))
        plt.suptitle("Audio frequency spectrum", fontsize=16)
        plt.specgram(self.l_channel, Fs=self.sampling_rate, vmin=-20, vmax=40)
        plt.xlabel("Duration (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar()
        plt.show()

    def transcribe(self) -> list[Word]:
        """Transcribe audio and save transcipt to .csv file

        Returns:
            list[Word]: List of transcripted words
        """
        model_path = self.repo_path.joinpath("models/vosk-model-small-pl-0.22")
        model = Model(str(model_path))
        list_of_words = []

        if self.r_channel is not None:
            self._save_audio_mono()
        else:
            self.audio_mono_file_path = self.audio_path

        with wave.open(str(self.audio_mono_file_path), "rb") as wf:
            rec = KaldiRecognizer(model, wf.getframerate())
            rec.SetWords(True)
            results = []
            # recognize speech using vosk model
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    part_result = json.loads(rec.Result())
                    results.append(part_result)

            part_result = json.loads(rec.FinalResult())
            results.append(part_result)

            for word in results:
                if len(word) == 1:
                    # sometimes there are bugs in recognition
                    # and it returns an empty dictionary
                    # {'text': ''}
                    continue
                for obj in word["result"]:
                    w = Word(obj)  # create custom Word object
                    list_of_words.append(w)  # and add it to list

        with open(self.transcript_file_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(("word", "start_time", "end_time", "confidence"))
            for word in list_of_words:
                writer.writerow(word.to_tuple())
        return list_of_words

    def cut_word(self, word: str):
        df_words = pd.read_csv(self.transcript_file_path, index_col="word")
        start_time = df_words["start_time"].get(word)[0]
        end_time = df_words["end_time"].get(word)[0]
        start_sample, stop_sample = ut.get_sample_index_by_time(
            start_time, end_time, self.sampling_rate
        )
        print(start_sample)
        print(stop_sample)
        audio_cut = ut.cut_audio_segment(start_sample, stop_sample, self.l_channel)

        audio_cut_file = self.regx.sub(f"_no_{word}.wav", self.audio_file)
        audio_cut_file_path = self.audio_dir.joinpath(audio_cut_file)

        wavfile.write(
            audio_cut_file_path,
            self.sampling_rate,
            audio_cut.astype(self.audio_data_type),
        )


def main():
    wavy = Wavy("audio.wav")
    print(repr(wavy))
    print(str(wavy))

    wavy.cut_word("ja")

    list_of_words = wavy.transcribe()
    for word in list_of_words:
        print(word.to_string())

    wavy.plot_audio_amplitude(db_scale=False)
    wavy.plot_audio_frequency_spectrum()

    wavy.remove_silence_by_signal_aplitude_threshold(
        threshold=0.1,
        verbose=True,
        plot=False,
    )

    # wavy.remove_silence_by_db_threshold(
    #     db_threshold=30,
    #     min_duration=0.1,
    #     verbose=True,
    #     plot=True,
    # )
    return 0


if __name__ == "__main__":
    sys.exit(main())
