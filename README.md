# Running the pipeline

Open a terminal in the directory 'DL_NLP' and follow the steps below.

**Step 1: preprocessing**
First creates a vocabulary from the trainings data containing word2index and index2word dictionaries.
Next this vocabulary is used to build indices vectors for all training, test and validation sentences.
The preprocessing input files are stored in the directory 'data/input', 
the preprocessing output files are written to the directory 'data/preprocess'.

```console
$ python preprocess.py
```

**Step 2: training**

**Step 3: prediction**

**Step 4: evaluation**

# Running the unit tests

Open a terminal in the directory 'DL_NLP' and run the command

```console
$ python -m unittest discover -v
```

