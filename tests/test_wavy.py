# Standard Library Imports
import unittest

# Third-party Imports
import pandas as pd

# Local Imports
from src.wavy import Wavy
from src.tools.utils import get_repo_path


class TestWavy(unittest.TestCase):
    def setUp(self) -> None:
        repo_path = get_repo_path()
        audio_dir = repo_path.joinpath("audio")

        audio_file = "test.wav"
        self.audio_filename = audio_dir.joinpath(audio_file)
        self.wavy = Wavy(audio_file)

    def test_incorrect_file_path_provided(self):
        with self.assertRaises(FileNotFoundError):
            Wavy("incorrect_file_path")

    def test_repr(self):
        wavy_formal_string = f"<class '{self.wavy.__class__.__name__}' from audio file: {self.wavy.audio_path}>"
        self.assertEqual(wavy_formal_string, repr(self.wavy))

    def test_str(self):
        wavy_informal_string = str(
            f"{'#'*20} AUDIO {'#'*20}\n"
            f"|Audio file:         {str(self.wavy.audio_path)}\n"
            f"|Data type:          int16\n"
            f"|Shape:              (57600, 2)\n"
            f"|Sampling rate:      48000\n"
            f"|Duration:           1.2s\n"
            f"|Min amplitude:      0\n"
            f"|Max amplitude:      2191\n"
            f"{'#'*60}"
        )
        self.assertEqual(wavy_informal_string, str(self.wavy))

    def test_remove_silence_by_signal_aplitude_threshold(self):
        threshold = 5
        wavy_no_silence = Wavy(
            self.wavy.remove_silence_by_signal_aplitude_threshold(threshold)
        )
        self.assertLessEqual(threshold, wavy_no_silence.min_amplitude)

    def test_transcribe(self):
        wavy_list_of_words = self.wavy.transcribe()
        wavy_list_of_words = [w.to_list() for w in wavy_list_of_words]
        transcript_list_of_words = pd.read_csv(self.wavy.transcript_file_path)
        transcript_list_of_words = transcript_list_of_words.values.tolist()
        self.assertEqual(transcript_list_of_words, wavy_list_of_words)


if __name__ == "__main__":
    unittest.main()
