# Standard Library Imports
import re
import unittest

# Local Imports
from src.wavy import Wavy
from src.utils import get_repo_path


class TestWavy(unittest.TestCase):
    def setUp(self) -> None:
        repo_path = get_repo_path()
        audio_dir = repo_path.joinpath("audio")

        audio_file = "test.wav"
        self.audio_filename = audio_dir.joinpath(audio_file)
        self.wavy = Wavy(audio_file)

    def test_incorrect_init(self):
        with self.assertRaises(FileNotFoundError):
            Wavy("incorrect_file_path")

    def test_repr(self):
        wavy_formal_string = f"<Wavy class of audio file: {self.audio_filename}>"
        self.assertEqual(wavy_formal_string, repr(self.wavy))

    def test_str(self):
        wavy_informal_string = str(
            "######################## AUDIO ########################\n"
            f"|Audio file:         {str(self.audio_filename)}\n"
            f"|Data type:          int16\n"
            f"|Shape:              (57600, 2)\n"
            f"|Sampling rate:      48000\n"
            f"|Duration:           1.2s\n"
            f"|Min amplitude:      0\n"
            f"|Max amplitude:      2191\n"
            "#######################################################"
        )
        self.assertEqual(wavy_informal_string, str(self.wavy))

    def test_remove_silence(self):
        threshold = 5
        self.wavy.remove_silence(threshold)
        regx = re.compile(".wav\\b")
        wavy_no_silence = Wavy(regx.sub("_no_silence.wav", str(self.audio_filename)))
        self.assertLessEqual(threshold, wavy_no_silence.min_amplitude)


if __name__ == "__main__":
    unittest.main()
