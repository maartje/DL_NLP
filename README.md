# Installation

* install pytorch, matplotlib, ...

# Running the unit tests

Open a terminal in the directory 'DL_NLP' and run the command

```console
$ python -m unittest discover -v
```

# Running the pipeline

Open a terminal in the directory 'DL_NLP' and follow the steps below.

**Step 1: create directory structure and prepare input files**

```console
$ ./prepare.sh
```

The './prepare.sh' script creates a directory structure that is assumed by the subsequent steps.

**Step 2: Dowload input files**

Download the data from https://zenodo.org/record/841984/files/wili-2018.zip?download=1 
and copy the train and test files to the directory 'data/wili-2018' that is created in the previous step.

**Step 3: preprocessing, training, prediction and evaluation**

To run the full pipeline, run the command:

```console
$ python pipeline.py
```

Alternatively, run the subsequent steps one-by-one.

```console
$ python preprocess.py
$ python train.py
$ python predict.py
$ python tf_idf_baseline.py
$ python evaluate.py
```

The settings of a run can be configured
by adapting the file 'config.py'.

The 'preprocess.py' script build indices vectors 
for all training and test sentences and labels.
It also creates dictionaries
that define the mapping between tokens and indices.
We currently use characters as tokens.
The preprocessing output files are written to the directory 'data/preprocess'.

The 'train.py' script trains a RNN (or CNN) model for language recognition.
The model and losses per epoch data are written to the
directory 'data/train'.

The 'predict.py' script creates the predicted probabilities for the test
(and train) dataset using the trained model.
For convenience the predicted probabilities are stored
together with the target values and the sentence lengths.
The output files are in 'data/predict'.

The tf_idf_baseline.py script runs a naive bayes model
using word or character counts as features. The model
is evaluated for different sequence lengths ranging from
0 to max_seq_length specified in the config

The 'evaluate.py' produces evaluation output used in the report:
- plots such as: train/validation loss per epoch, 
                 validation accuracy per epoch, 
                 test accuracy per character position,
                 comparison of different models
- confusion matrix
- scores for: train/test loss, average train/test accuracy
The output files are stored in 'data/evaluate'.


