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
import librosa
import matplotlib.pyplot as plt
from vosk import Model, KaldiRecognizer
from scipy.io import wavfile

# Local Imports
import src.tools.utils as ut
from src.tools.word import Word


class Wavy:
    def __init__(self, audio_file: str) -> None:
        self.audio_file = audio_file
        self.regx = re.compile(".wav\\b")
        self.repo_path = ut.get_repo_path()
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
            # audio_normalized = self.audio / (2.0**15)

            # db_audio = np.vectorize(ut.convert_to_decibel)(
            # self.audio / (2.0**15)
            # )
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

    def _save_audio_mono(self) -> None:
        audio_mono_file = self.regx.sub("_mono.wav", self.audio_file)
        self.audio_mono_file_path = self.audio_dir.joinpath(audio_mono_file)
        wavfile.write(
            self.audio_mono_file_path,
            self.sampling_rate,
            self.l_channel.astype(self.audio_data_type),
        )

    def _remove_silence_plot(
        self,
        time_arrays: tuple,
        audio_arrays: tuple,
        silence_samples_indexes: np.ndarray,
    ) -> None:
        fig, axs = plt.subplots(3, 1)
        fig.suptitle("Audio amplitude", fontsize=16)
        fig.set_figheight(10)
        fig.set_figwidth(16)
        axs[0].plot(time_arrays[0], audio_arrays[0], color="r", label="original audio")
        axs[0].plot(
            time_arrays[0][silence_samples_indexes],
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

    def remove_silence_by_signal_aplitude_threshold(
        self, threshold=4, plot=False
    ) -> None:
        silence_samples_indexes = np.where(np.abs(self.l_channel) < threshold)[0]
        silence_samples_count = self.l_channel[silence_samples_indexes].shape[0]

        no_silence_samples_indexes = np.where(np.abs(self.l_channel) >= threshold)[0]
        no_silence_samples_count = self.l_channel[no_silence_samples_indexes].shape[0]

        og_d = math.modf(np.round(self.audio_samples / self.sampling_rate / 60, 2))
        s_d = math.modf(np.round(silence_samples_count / self.sampling_rate / 60, 2))
        n_s_d = math.modf(
            np.round(no_silence_samples_count / self.sampling_rate / 60, 2)
        )

        audio_no_silence_file = self.regx.sub("_no_silence.wav", self.audio_file)
        audio_no_silence_file_path = self.audio_dir.joinpath(audio_no_silence_file)

        print("################## Removing silence ###################")
        print(f"|Audio without silence file: {audio_no_silence_file_path}")
        print(f"|")
        print(f"|Original samples: {self.audio_samples}")
        print(f"|Original duration: {int(og_d[1])}min {int(60 * og_d[0])}s")
        print(f"|")
        print(f"|No silence samples: {no_silence_samples_count}")
        print(f"|No silence duration: {int(n_s_d[1])}min {int(60 * n_s_d[0])}s")
        print(f"|")
        print(f"|Silence samples: {silence_samples_count}")
        print(f"|Silence duration: {int(s_d[1])}min {int(60 * s_d[0])}s")
        print("#######################################################")

        wavfile.write(
            audio_no_silence_file_path,
            self.sampling_rate,
            self.audio[no_silence_samples_indexes].astype(self.audio_data_type),
        )

        if plot:
            time_array = ut.calculate_time_array(self.audio_samples, self.sampling_rate)
            silence_time_array = ut.calculate_time_array(
                silence_samples_count, self.sampling_rate
            )
            no_silence_time_array = ut.calculate_time_array(
                no_silence_samples_count, self.sampling_rate
            )
            time_arrays = (time_array, no_silence_time_array, silence_time_array)
            audio_arrays = (
                self.audio,
                self.audio[no_silence_samples_indexes],
                self.audio[silence_samples_indexes],
            )
            self._remove_silence_plot(
                time_arrays, audio_arrays, silence_samples_indexes
            )

    def remove_silence_by_db_threshold(
        self, db_threshold=30, min_duration=0.1, plot=False
    ) -> None:
        min_samples = int(min_duration * self.sampling_rate)
        non_silent_intervals = librosa.effects.split(
            self.l_channel,
            top_db=db_threshold,
            frame_length=min_samples,
            hop_length=min_samples,
        )
        print(non_silent_intervals)
        silence_samples_indexes = np.where(self.db_audio_l < -db_threshold)[0]

        diff = np.diff(silence_samples_indexes)  # [:-1] - silence_samples_indexes[1:]
        diff = np.append(diff, 1)
        print(diff)
        diff = ut.find_subarrays(diff, np.full(min_samples, 1))

        si = np.array([])
        for i in range(int(len(silence_samples_indexes) / min_samples)):
            if i * min_samples + min_samples > len(diff):
                continue
            if (
                diff[i * min_samples + min_samples] - diff[i * min_samples]
                == min_samples
            ):
                si = np.concatenate(
                    (si, diff[i * min_samples : i * min_samples + min_samples])
                )

        silence_samples_indexes = silence_samples_indexes[si.astype(int)]
        silence_samples_count = self.l_channel[silence_samples_indexes].shape[0]

        no_silence_samples_indexes = np.setdiff1d(
            np.argwhere(self.l_channel),
            np.argwhere(self.l_channel[silence_samples_indexes]),
        )
        no_silence_samples_count = self.l_channel[no_silence_samples_indexes].shape[0]

        og_d = math.modf(np.round(self.audio_samples / self.sampling_rate / 60, 2))
        s_d = math.modf(np.round(silence_samples_count / self.sampling_rate / 60, 2))
        n_s_d = math.modf(
            np.round(no_silence_samples_count / self.sampling_rate / 60, 2)
        )

        audio_no_silence_file = self.regx.sub("_no_silence.wav", self.audio_file)
        audio_no_silence_file_path = self.audio_dir.joinpath(audio_no_silence_file)

        print("################## Removing silence ###################")
        print(f"|Audio without silence file: {audio_no_silence_file_path}")
        print(f"|")
        print(f"|Original samples: {self.audio_samples}")
        print(f"|Original duration: {int(og_d[1])}min {int(60 * og_d[0])}s")
        print(f"|")
        print(f"|No silence samples: {no_silence_samples_count}")
        print(f"|No silence duration: {int(n_s_d[1])}min {int(60 * n_s_d[0])}s")
        print(f"|")
        print(f"|Silence samples: {silence_samples_count}")
        print(f"|Silence duration: {int(s_d[1])}min {int(60 * s_d[0])}s")
        print("#######################################################")

        wavfile.write(
            audio_no_silence_file_path,
            self.sampling_rate,
            self.audio[no_silence_samples_indexes].astype(self.audio_data_type),
        )

        if plot:
            time_array = ut.calculate_time_array(self.audio_samples, self.sampling_rate)
            silence_time_array = ut.calculate_time_array(
                silence_samples_count, self.sampling_rate
            )
            no_silence_time_array = ut.calculate_time_array(
                no_silence_samples_count, self.sampling_rate
            )
            time_arrays = (time_array, no_silence_time_array, silence_time_array)
            audio_arrays = (
                self.db_audio_l,
                self.db_audio_l[no_silence_samples_indexes],
                self.db_audio_l[silence_samples_indexes],
            )
            self._remove_silence_plot(
                time_arrays, audio_arrays, silence_samples_indexes
            )

    def plot_audio_amplitude(self, db_scale=False) -> None:
        """Plot orginal audio signal amplitude"""
        if self.r_channel is not None:
            fig, axs = plt.subplots(3, 1)
            fig.suptitle("Audio amplitude", fontsize=16)
            fig.set_figheight(10)
            fig.set_figwidth(16)
            if not db_scale:
                axs[0].plot(
                    ut.calculate_time_array(self.audio_samples, self.sampling_rate),
                    self.l_channel,
                    color="r",
                )
            else:
                axs[0].plot(
                    ut.calculate_time_array(self.audio_samples, self.sampling_rate),
                    self.db_audio_l,
                    color="r",
                )
            axs[0].set_title("Left channel")
            axs[0].set_ylabel("Amplitude")
            if not db_scale:
                axs[1].plot(
                    ut.calculate_time_array(self.audio_samples, self.sampling_rate),
                    self.r_channel,
                    color="g",
                )
            else:
                axs[1].plot(
                    ut.calculate_time_array(self.audio_samples, self.sampling_rate),
                    self.db_audio_r,
                    color="g",
                )
            axs[1].set_title("Right channel")
            axs[1].set_ylabel("Amplitude")
            if not db_scale:
                axs[2].plot(
                    ut.calculate_time_array(self.audio_samples, self.sampling_rate),
                    self.r_channel - self.l_channel,
                    color="b",
                )
            else:
                axs[2].plot(
                    ut.calculate_time_array(self.audio_samples, self.sampling_rate),
                    self.db_audio_r - self.db_audio_l,
                    color="b",
                )
            axs[2].set_title("Difference between channels")
            axs[2].set_ylabel("Amplitude")
            axs[2].set_xlabel("Duration (s)")
        else:
            plt.figure(figsize=(16, 10))
            plt.title("Audio amplitude", fontsize=16)
            if not db_scale:
                plt.plot(
                    ut.calculate_time_array(self.audio_samples, self.sampling_rate),
                    self.l_channel,
                    color="r",
                )
            else:
                plt.plot(
                    ut.calculate_time_array(self.audio_samples, self.sampling_rate),
                    self.db_audio_l,
                    color="r",
                )
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

    def transcribe(self) -> list:
        model_path = self.repo_path.joinpath("models/vosk-model-small-pl-0.22")
        model = Model(str(model_path))
        list_of_words = []

        if self.r_channel is not None:
            self._save_audio_mono()
            audio_path = str(self.audio_mono_file_path)
        else:
            audio_path = str(self.audio_path)

        with wave.open(audio_path, "rb") as wf:
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


def main():
    test_wavy = Wavy("test.wav")
    print(repr(test_wavy))
    print(test_wavy)

    # test_wavy.plot_audio_amplitude(db_scale=False)
    test_wavy.remove_silence_by_signal_aplitude_threshold(threshold=5)
    test_wavy.remove_silence_by_db_threshold(db_threshold=30, min_duration=0.1)
    # test_wavy.plot_audio_frequency_spectrum()

    list_of_words = test_wavy.transcribe()
    for word in list_of_words:
        print(word.to_string() + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
