---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.7
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Based on our custom MIMIC language model, train a classifier

Make sure mimic_nlp_lm has been run first and sucessfully completed. That notebook builds the language model that allows classificiation to occur.

```python
from fastai.text import *
from sklearn.model_selection import train_test_split
import glob
import gc
```

```python
class_file = 'mimic_cl.pickle'
filename = base_path/class_file

if os.path.isfile(filename):
    data_cl = load_data(base_path, file, bs=bs)
else:
    data_cl = (TextList.from_df(df, cols='', vocab=data_lm.vocab)
               #grab all the text files in path
               .split_by_folder(valid='test')
               #split by train and valid folder (that only keeps 'train' and 'test' so no need to filter)
               .label_from_folder(classes=['neg', 'pos'])
               #label them all with their folders
               .databunch(bs=bs))

data_cl.save(filename)
```

```python
df.head()
```

```python
data_cl.show_batch()
```

```python
len(df.CATEGORY.unique())
```

```python
len(df.DESCRIPTION.unique())
```

```python
if os.path.isfile(filename):
    data_lm = load_data(base_path, file, bs=bs)
else:
    data_lm = (TextList.from_df(df, 'texts.csv', cols='TEXT')
               #df has several columns; actual text is in column TEXT
               .split_by_rand_pct(valid_pct=valid_pct, seed=seed)
               #We randomly split and keep 10% for validation
               .label_from_df(cols='DESCRIPTION')
               #We want to do a language model so we label accordingly
               .databunch(bs=bs))
    data_lm.save(filename)
```

<!-- #region -->

This is the version from the original example


```python
data_clas = (TextList.from_folder(path, vocab=data_lm.vocab)
             #grab all the text files in path
             .split_by_folder(valid='test')
             #split by train and valid folder (that only keeps 'train' and 'test' so no need to filter)
             .label_from_folder(classes=['neg', 'pos'])
             #label them all with their folders
             .databunch(bs=bs))

data_clas.save('data_clas.pkl')
```
<!-- #endregion -->

```python
learn = text_classifier_learner(data_cl, AWD_LSTM, drop_mult=0.5)
learn.load_encoder(enc_file)
```

```python
learn.lr_find()
```

```python
learn.recorder.plot()
```

Change learning rate based on results from the above plot

```python
learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
```
