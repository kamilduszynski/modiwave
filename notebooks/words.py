# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import path  # nopycln: import

# Standard Library Imports
import json
import wave
from collections import Counter

# Third-party Imports
import pandas as pd
from vosk import Model, KaldiRecognizer

# Local Imports
from word import Word
from utils import get_repo_path

# %%
repo_path = get_repo_path()
model_path = repo_path + "models/vosk-model-small-pl-0.22"
audio_filename = repo_path + "audio/audio_mono.wav"  # _no_silence
model = Model(model_path)

# %%
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

# %%
results = Counter()
df_words = pd.DataFrame([word.to_dict() for word in list_of_words])
df_words["word"].apply(lambda x: results.update(x.split()))

results = [{"word": word, "count": count} for word, count in results.items()]
df_words_count = pd.DataFrame(results).sort_values(by="count", ascending=False)
# %%
df_words_count.head(20)
# %%
