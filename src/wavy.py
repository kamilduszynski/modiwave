# Standard Library Imports
import sys

# Third-party Imports
import numpy as np
from scipy.io import wavfile

# Local Imports
from src.utils import get_repo_path


class Wavy:
    def __init__(self, audio_file: str):
        self.repo_path = get_repo_path()
        self.audio_dir = self.repo_path.joinpath("audio")
        self.audio_filename = self.audio_dir.joinpath(audio_file)

        if self.audio_filename.exists():
            self.sampling_freq, self.audio = wavfile.read(str(self.audio_filename))
            self.audio_data_type = self.audio.dtype
            self.audio_shape = self.audio.shape
            self.max_amplitude = np.abs(self.audio).max()
            self.min_amplitude = np.abs(self.audio).min()

            self.channel_0 = self.audio[:, 0]
            if self.audio.shape[1] == 2:
                self.channel_1 = self.audio[:, 0]
        else:
            raise FileNotFoundError("Provided audio file does not exist")

    def __repr__(self):
        wavy_formal_string = str(f"<Wavy class of audio file: {self.audio_filename}>")
        return wavy_formal_string

    def __str__(self):
        wavy_informal_string = str(
            "################ AUDIO ######################\n"
            f"|Audio file:         {str(self.audio_filename)}\n"
            f"|Data type:          {self.audio_data_type}\n"
            f"|Shape:              {self.audio_shape}\n"
            f"|Sampling frequency: {self.sampling_freq}\n"
            f"|Min amplitude:      {self.min_amplitude}\n"
            f"|Max amplitude:      {self.max_amplitude}\n"
            "#############################################"
        )
        return wavy_informal_string


def main():
    test_wavy = Wavy("test.wav")
    print(repr(test_wavy))
    print(test_wavy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
