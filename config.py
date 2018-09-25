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
    'targets_train' : 'data/preprocess/targets_train_indices.pt',
    'targets_test' : 'data/preprocess/targets_test_indices.pt',
    'targets_dictionairies' : 'data/preprocess/targets_dictionairies.pt', 

    # output train
    'epoch_metrics' : 'data/train/epoch_metrics.pt',
    'model' : 'data/train/model.pt',

    # output predict
    'predictions_train' : 'data/predict/test.pt',
    'predictions_test'  : 'data/predict/train.pt',

    # output evaluate
    'plot_epoch_losses' : 'data/evaluate/plot_epoch_losses.png',
    'plot_epoch_accuracies' : 'data/evaluate/plot_epoch_accuracies.png',
    'plot_accuracy_seq_length' : 'data/evaluate/plot_seq_acc.png'
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