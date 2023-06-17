# Standard Library Imports
import sys
import json
import wave

# Third-party Imports
from vosk import Model, KaldiRecognizer

# Local Imports
from word import Word
from utils import get_repo_path


def get_list_of_words(audio_filename, model):
    with wave.open(audio_filename, "rb") as wf:
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

        # convert list of JSON dictionaries to list of 'Word' objects
        list_of_words = []
        for sentence in results:
            if len(sentence) == 1:
                # sometimes there are bugs in recognition
                # and it returns an empty dictionary
                # {'text': ''}
                continue
            for obj in sentence["result"]:
                w = Word(obj)  # create custom Word object
                list_of_words.append(w)  # and add it to list
    return list_of_words


def main() -> int:
    repo_path = get_repo_path()
    audio_file = repo_path.joinpath("audio/audio_mono.wav")
    model_path = repo_path.joinpath("models/vosk-model-small-pl-0.22")
    model = Model(str(model_path))

    list_of_words = get_list_of_words(str(audio_file), model)

    with open("audio/trascript.txt", "w") as f:
        for word in list_of_words:
            print(word.to_string())
            f.write(f"{word.to_string()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
