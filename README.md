# Tasks

* Implement basic pipeline (for RNN model) [Maartje]
* Implement a script to select texts that share the same character set 
(for now it is enough to select all latin character texts by target language)
* Find literature on neural language recognition

# Running the pipeline

Open a terminal in the directory 'DL_NLP' and follow the steps below.

**Step 1: create directory structure and prepare input files**

```console
$ ./prepare.sh
```

The './prepare.sh' script creates a directory structure that is assumed by the subsequent steps.

After running the script, copy the input data that you want use 
to the directory 'data/input'

**Step 2: preprocessing**

```console
$ python preprocess.py
```

This script first creates a vocabulary from the trainings data containing 
word2index and index2word dictionaries.
Next this vocabulary is used to build indices vectors 
for all training, test and validation sentences.

The preprocessing output files are written to the directory 'data/preprocess'.

**Step 3: training**

```console
$ python train.py
```
work in progress ...

**Step 4: prediction**

```console
$ python predict.py
```

work in progress ...

**Step 5: evaluation**

```console
$ python evaluate.py
```

work in progress ...

# Running the unit tests

Open a terminal in the directory 'DL_NLP' and run the command

```console
$ python -m unittest discover -v
```

