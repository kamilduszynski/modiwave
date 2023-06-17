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
from collections import Counter

# Third-party Imports
import pandas as pd
from scipy.io import wavfile
from vosk import Model

# Local Imports
from transcribe import get_list_of_words
from utils import get_repo_path, get_sample_index_by_time

# %%
repo_path = get_repo_path()
model_path = repo_path.joinpath("models/vosk-model-small-pl-0.22")
audio_filename = repo_path.joinpath("audio/audio_mono.wav")
model = Model(str(model_path))

# %%
sampling_freq, audio = wavfile.read(str(audio_filename))
list_of_words = get_list_of_words(audio_filename, model)

# %%
results = Counter()
df_words = pd.DataFrame([word.to_dict() for word in list_of_words])
df_words["word"].apply(lambda x: results.update(x.split()))

results = [{"word": word, "count": count} for word, count in results.items()]
df_words_count = pd.DataFrame(results)
# %%
df_words_count.sort_values(by="count", ascending=True).head(50)
# %%
df_words_count

# %%
df_words

# %%
df_words["start_sample"], df_words["end_sample"] = df_words.apply(
    lambda x: get_sample_index_by_time(x["start_time"], x["end_time"], sampling_freq),
    axis=1,
)

# %%
output = df_words.apply(lambda x: print(x["start_time"], x["end_time"]), axis=1)
# %%

# %%
