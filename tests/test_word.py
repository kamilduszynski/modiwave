# Standard Library Imports
import unittest

# Local Imports
from src.tools.word import Word


class TestWord(unittest.TestCase):
    def setUp(self) -> None:
        self.test_word = Word({"word": "test", "start": 0, "end": 1, "conf": 0.99})

    def test_to_string(self):
        word_string = (
            "{:20} from {:.2f} sec to {:.2f} sec, confidence is {:.2f}%".format(
                self.test_word.word,
                self.test_word.start,
                self.test_word.end,
                self.test_word.conf,
            )
        )
        self.assertEqual(self.test_word.to_string(), word_string)

    def test_to_dict(self):
        word_dict = {"word": "test", "start": 0, "end": 1, "conf": 0.99 * 100}
        self.assertEqual(self.test_word.to_dict(), word_dict)

    def test_to_tuple(self):
        word_list = ("test", 0, 1, 0.99 * 100)
        self.assertEqual(self.test_word.to_tuple(), word_list)

    def test_to_list(self):
        word_list = ["test", 0, 1, 0.99 * 100]
        self.assertEqual(self.test_word.to_list(), word_list)
