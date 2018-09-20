filepaths = {
    # directories
    'dir_input' : 'data/input',
    'dir_preprocess' : 'data/preprocess',
    'dir_train' : 'data/train',
    'dir_predict' : 'data/predict',
    'dir_evaluate' : 'data/evaluate',

    # input files

    # output preprocess
    'vocab' : 'data/preprocess/vocab.pt',
    'vectors_train' : 'data/preprocess/vectors_train.pt',
    'vectors_test' : 'data/preprocess/vectors_test.pt',
    'targets_train' : 'data/preprocess/targets_train.pt',
    'targets_test' : 'data/preprocess/targets_test.pt',
}

settings = {
    # preprocess
    'min_occurrence' : 2,
    'PAD_index' : 0,

    # train
    'batch_size' : 128

    # predict

    # evaluate
}