---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

## Using FAST.AI for NLP

Exploring the MIMIC III data set medical notes.

Tried working with the full dataset, but almost every training step takes many hours (~13 for initial training), predicted 14+ per epoch for fine tuning.

Instead will try to work with just 10% sample... Not sure that will work though

A few notes:
* See https://docs.fast.ai/text.transform.html#Tokenizer for details on what various artificial tokens (e.g xxup, xxmaj, etc.) mean
* Due to a change in the markdown package private API, the 'doc' functionality (e.g. ` doc(learn.lr_find)`) is currently broken. See https://github.com/fastai/fastai/commit/21faa5d187b2cccf2a48315d183c2863ed2cdc50

```python
from fastai.text import *
from sklearn.model_selection import train_test_split
```

```python
# run this to see what has already been imported
#whos
```

```python
# pandas doesn't understand ~, so provide full path
base_path = Path('/home/jupyter/mimic')
```

```python
# run this the first time to covert CSV to Pickle file
df = pd.read_csv(base_path/'NOTEEVENTS.csv', low_memory=False, memory_map=True)
df.to_pickle(base_path/'noteevents.pickle')
```

```python
filename = base_path/'noteevents.pickle'

if os.path.isfile(filename):
    # this is much faster than reading a csv
    orig_df = pd.read_pickle(filename)
else:
    print('Could not find noteevent pickle file; creating it')
    # run this the first time to covert CSV to Pickle file
    orig_df = pd.read_csv(base_path/'NOTEEVENTS.csv', low_memory=False, memory_map=True)
    orig_df.to_pickle(filename)
```

```python
df = orig_df.sample(frac=0.1)
```

```python
df.head()
```

```python
df.dtypes
```

```python
df.shape
```

```python
# split data into train and test sets
seed = 42
test_size = 0.333333333
train, test = train_test_split(df, test_size=test_size, random_state=seed)
```

```python
train.shape
```

```python
test.shape
```

```python
# previously used 48; worked fine but never seemed to use even half of GPU memory
bs=64
```

<!-- #region -->
Code to reload previously built language model

```python
filename = base_path/'mimic_lm.pickle'

if os.path.isfile(filename):
    data_lm = load_data(base_path, 'mimic_lm.pickle', bs=bs)
else:
    print('Couldnt find file')
```
<!-- #endregion -->

<!-- #region -->
Code to build initial version of language model

```python
```
<!-- #endregion -->

```python
## why does this only seem to use CPU?
# both textclasdatabunch and textlist...
# run out of memory at 32 GB, error at 52 GB, trying 72GB now... got down to only 440MB free; if crash again, increase memory
# now at 20vCPU and 128GB RAM; ok up to 93%; got down to 22GB available
# succeeded with 20CPU and 128GB RAM...
# try smaller batch size? will that reduce memory requirements?
data_lm = (TextList.from_df(df, 'texts.csv', cols='TEXT')
           #We may have other temp folders that contain text files so we only keep what's in train and test
           .split_by_rand_pct(0.1)
           #We randomly split and keep 10% for validation
           .label_for_lm()
           #We want to do a language model so we label accordingly
           .databunch(bs=bs))
```

```python
data_lm.save(base_path/'mimic_lm.pickle')
```

```python
data_lm = load_data(base_path, 'mimic_lm.pickle', bs=bs)
```

<!-- #region -->
If need to view more data, run appropriate line to make display wider/show more columns...
```python
# default 20
pd.get_option('display.max_columns')
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_columns', None) # show all
# default 50
pd.get_option('display.max_colwidth')
pd.set_option('display.max_colwidth', -1) # show all
```
<!-- #endregion -->

```python
data_lm.show_batch()
# how to look at original version of text
#df[df['TEXT'].str.contains('being paralyzed were discussed', case=False)].TEXT
```

```python
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
learn.lr_find()
```

```python
learn.recorder.plot(skip_end=15)
```

```python
# no idea how long nor how much resources this will take
# not sure 1e-2 is the right learning rate; maybe 1e-1 or between 1e-2 and 1e-1
# using t4
# progress bar says this will take around 24 hours... ran for about 52 minutes
# gpustat/nvidia-smi indicates currently only using about 5GB of GPU RAM
# using p100
# progress bar says this will take around 12 hours; took 13:16
# at start GPU using about 5GB RAM
# after about 8 hours GPU using about 7.5GB RAM.
# looks like I could increase batch size...
learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))
```

```python
learn.save(base_path/'mimic_fit_head.pickle')
```

```python
learn.load(base_path/'mimic_fit_head.pickle')
```

```python
learn.unfreeze()
```

```python
learn.fit_one_cycle(10, 1e-3, moms=(0.8,0.7))
```

```python
learn.save(base_path/'mimic_fine_tuned.pickle')
```

```python
learn.load(base_path/'mimic_fine_tuned.pickle')
```

```python
# test the language generation capabilities of this model (not the point, but is interesting)
TEXT = "For confirmation, she underwent CTA of the lung which was negative for pulmonary embolism"
N_WORDS = 40
N_SENTENCES = 2
In [ ]:
print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))
```

```python
learn.save_encoder('mimic_fine_tuned_enc.pickle')
```
