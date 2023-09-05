# Standard Library Imports
import sys


class Word:
    """Class representing a word from the JSON format for vosk speech recognition API"""

    def __init__(self, word_dict: dict):
        """
        Parameters:
          word_dict (dict) dictionary from JSON, containing:
            conf (float): degree of confidence, from 0 to 1
            end (float): end time of the pronouncing the word, in seconds
            start (float): start time of the pronouncing the word, in seconds
            word (str): recognized word
        """
        self.word = word_dict["word"]
        self.start = word_dict["start"]
        self.end = word_dict["end"]
        self.conf = word_dict["conf"] * 100

    def to_string(self) -> str:
        """Returns a string describing this instance"""
        return "{:20} from {:.2f} sec to {:.2f} sec, confidence is {:.2f}%".format(
            self.word, self.start, self.end, self.conf
        )

    def to_dict(self) -> dict:
        """Returns a dict describing this instance"""
        return self.__dict__

    def to_tuple(self) -> tuple:
        """Returns a tuple describing this instance"""
        return tuple(self.__dict__.values())

    def to_list(self) -> list:
        """Returns a list describing this instance"""
        return list(self.__dict__.values())


def main():
    test_word = Word({"word": "test", "start": 0, "end": 1, "conf": 0.99})
    print(test_word.to_string())
    print(test_word.to_dict())
    print(test_word.to_tuple())
    print(test_word.to_list())


if __name__ == "__main__":
    sys.exit(main())
