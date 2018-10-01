# Tasks

* plot **confusion matrix**: evaluate.py, src/reporting/metrics (for the calculation), src/reporting/plots.py
* specify two **language filters** in config.py (e.g. latin- and chinese- character languages, see language_filters['test'] for an example)
* **train** the model (preferable on GPU) for the latin character languages and **experiment with hyper parameters** (learning rate, max_seq_length, ...) [Basak]

* Find **scientific papers** on neural language recognition

* experiment with using **words instead of characters** (even though they are not easy to compare). This basically means that you have to plug in another tokenizer (separate by space instead of just split all characters.). [Basak]

* Experiment with **alternative models**: may be a feed forward NN or a CNN with a fixed sequence length. That is: predict the langage after exactly n characters.

# DONE
* Setup the pipeline [Maartje]
* Implement preprocess for texts [Maartje]
* Implement RNN model [Maartje]
* Implement dataloading (for RNN model) [Maartje]
* Implement train and predict [Maartje]
* collect validation loss and save best model during training [Maartje]
* Plot train and validation loss [Maartje]
* Filter on language character group [Maartje]
* implement preprocessing labels [Kwesi]
* implement tf-idf with naive bayes as a **baseline model** and hang it in the pipeline. [Maartje]
* Implement **accuracy metric**: [Maartje]
    * calculate average accuracy per character position in 'src/reporting/metrics: calculate_accuracies'
    * collect average accuracy over all char positions during training (src/reporting/metrics_collector, see also validation loss)
    * use accuracy over validation set as criterion for selecting the best model in src/model_saver (instead of val_loss)
    * in evaluate.py and src/reporting/plots.py: 1. plot accuracies collected during training and 2. plot accuracy per character position for our best model

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

The 'train.py' script trains a RNN model for language recognition.
The model and losses per epoch data are written to the
directory 'data/train'.

The 'predict.py' script creates the predicted probabilities for the test
(and train) dataset using the trained model.
For convenience the predicted probabilities are stored
together with the target values and the sentence lengths.
The output files are in 'data/predict'.

The 'evaluate.py' produces evaluation output used in the report:
- plots such as: train/validation loss per epoch, 
                 validation accuracy per epoch, 
                 test accuracy per character position
- confusion matrix
- scores for: train/test loss, average train/test accuracy 
- ...
The output files are stored in 'data/evaluate'.


