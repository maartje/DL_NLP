filepaths = {
    # directories
    'dir_input' : 'data/input',
    'dir_preprocess' : 'data/preprocess',
    'dir_train' : 'data/train',
    'dir_predict' : 'data/predict',
    'dir_evaluate' : 'data/evaluate',

    # input files
    'labels_train' : 'data/wili-2018/y_train.txt',
    'labels_test' : 'data/wili-2018/y_test.txt',
    'texts_train' : 'data/wili-2018/x_train.txt',
    'texts_test' : 'data/wili-2018/x_test.txt',

    # output preprocess
    'vocab' : 'data/preprocess/vocab.pt',
    'vectors_train' : 'data/preprocess/vectors_train.pt',
    'vectors_test' : 'data/preprocess/vectors_test.pt',
    'targets_train' : 'data/preprocess/targets_train_indices.pickle',
    'targets_test' : 'data/preprocess/targets_test_indices.pickle',

    # output train
    'epoch_losses' : 'data/train/epoch_losses.pt',
    'model' : 'data/train/model.pt',

    # output predict
    'predictions_train' : 'data/predict/test.pt',
    'predictions_test'  : 'data/predict/train.pt'
}

settings = {
    # preprocess
    'language_filter' : 'test',
    'min_occurrence' : 2,
    'PAD_index' : 0, # DO NOT CHANGE THIS!

    # train
    'max_seq_length' : 25, 
    'rnn' : {
        'batch_size' : 4,
        'learning_rate' : 0.5,
        'epochs' : 10,
        'hidden_size' : 256,
        'drop_out' : 0.3
    }

    # predict

    # evaluate
}

language_filters = {
    'test' : ['nld', 'eng']
}