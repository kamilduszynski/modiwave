class Word:
    """Class representing a word from the JSON format for vosk speech recognition API"""

    def __init__(self, dict: dict):
        """
        Parameters:
          dict (dict) dictionary from JSON, containing:
            conf (float): degree of confidence, from 0 to 1
            end (float): end time of the pronouncing the word, in seconds
            start (float): start time of the pronouncing the word, in seconds
            word (str): recognized word
        """

        self.conf = dict["conf"]
        self.end = dict["end"]
        self.start = dict["start"]
        self.word = dict["word"]

    def to_string(self) -> str:
        """Returns a string describing this instance"""
        return "{:20} from {:.2f} sec to {:.2f} sec, confidence is {:.2f}%".format(
            self.word, self.start, self.end, self.conf * 100
        )

    def to_dict(self) -> dict:
        """Returns a dict describing this instance"""
        return {
            "word": self.word,
            "start_time": self.start,
            "end_time": self.end,
            "confidence": self.conf * 100,
        }
