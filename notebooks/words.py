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
# Standard Library Imports
from collections import Counter

# Third-party Imports
import path  # nopycln: import
import pandas as pd

# Local Imports
from wavy import Wavy
from tools.utils import get_sample_index_by_time

# %%
wavy = Wavy("test.wav")
list_of_words = wavy.transcribe()

# %%
df_words = pd.read_csv(wavy.transcript_file_path)

# %%
print(df_words.values.tolist())

# %%
results = Counter()
df_words["word"].apply(lambda x: results.update(x.split()))
results = [{"word": word, "count": count} for word, count in results.items()]
df_words_count = pd.DataFrame(results)
# %%
df_words_count.sort_values(by="count", ascending=False).head(50)
# %%
df_words_count

# %%
df_words["start_sample"], df_words["end_sample"] = df_words.apply(
    lambda x: get_sample_index_by_time(
        x["start_time"], x["end_time"], wavy.sampling_rate
    ),
    axis=1,
)

# %%
output = df_words.apply(lambda x: print(x["start_time"], x["end_time"]), axis=1)
# %%

# %%
