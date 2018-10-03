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
    'targets_dictionaries' : 'data/preprocess/targets_dictionaries.pt', 

    # output train
    'epoch_metrics' : 'data/train/epoch_metrics.pt',
    'model' : 'data/train/model.pt',

    # output predict
    'predictions_train' : 'data/predict/test.pt',
    'predictions_test'  : 'data/predict/train.pt',

    # output evaluate
    'naive_bayes_accuracies' : 'data/evaluate/naive_bayes_accuracies.pt',
    'plot_epoch_losses' : 'data/evaluate/plot_epoch_losses.png',
    'plot_epoch_accuracies' : 'data/evaluate/plot_epoch_accuracies.png',
    'plot_accuracy_seq_length' : 'data/evaluate/plot_seq_acc.png',
    'plot_accuracy_model_comparison' : 'data/evaluate/plot_accuracy_model_comparison.png',
    'plot_train_confusion_matrix':  'data/evaluate/plot_train_confusion_matrix.png',
    'plot_test_confusion_matrix':  'data/evaluate/plot_test_confusion_matrix.png'
}

settings = {
    # preprocess
    'model_name': 'rnn', #or rnn
    'model' : 'word', #or char
    'language_filter' : 'test',
    'min_occurrence' : 2,
    'PAD_index' : 0, # DO NOT CHANGE THIS!

    # train
    'val_train_ratio' : 0.1,
    'use_all_fragments' : True, # warning: if this setting is True and characters are used: each fragement results in many sub fragments (use GPU) 
    'max_seq_length' : 50, 
    'check_equal_seq_length' : False, # use True for now since padding is not supported yet in accuracy calculation
    'rnn' : {
        'batch_size' : 128,
        'learning_rate' : 0.005,
        'epochs' : 20,
        'hidden_size' : 256,
        'drop_out' : 0.3
    },
    'cnn': {
        'batch_size': 128,
        'learning_rate': 0.001,
        'epochs': 10,
        'hidden_size': 128,
        'drop_out': 0.5
    }

    # predict

    # evaluate
}

language_filters = {
    #'test'     : ['nld', 'eng'],
    'test'     : ['nld', 'eng', 'fra', 'ita', 'deu','afr', 'dan', 'fin', 'ltz', 'por', 'slk', 'swe', 'slv', 'gle', 'scn', 'spa', 'ron', 'lit', 'est','isl'],
    'latin'    : ['ace', 'afr', 'als', 'ang', 'arg', 'ast', 'aym', 'aze', 'bar', 'bcl', 'bjn',
                  'bre', 'cat', 'cbk', 'ceb', 'ces', 'cor', 'cos', 'csb', 'cym', 'dan', 'deu', 'diq', 'dsb',
                  'egl', 'eng', 'epo', 'est', 'eus', 'ext', 'fao', 'fin', 'fra', 'frp', 'fry', 'fur', 'gla',
                  'gle', 'glg', 'glv', 'grn', 'hat', 'hbs', 'hrv', 'hsb', 'hun', 'ibo', 'ido', 'ile', 'ilo', 
                  'ina', 'isl', 'ita', 'jam', 'jbo', 'kab', 'kin', 'ksh', 'lat', 'lav', 'lim', 'lin', 'lit', 
                  'lmo', 'ltg', 'ltz', 'lug', 'min', 'mlg', 'mlt', 'mri', 'mwl', 'nav', 'nci', 'nds', 'nds-nl',
                  'nld', 'nno', 'nob', 'nrm', 'nso', 'oci', 'olo', 'pag', 'pam', 'pap', 'pcd', 'pdc', 'pfl', 
                  'pol', 'por', 'que', 'roh', 'ron', 'rup', 'scn', 'sco', 'sgs', 'slk', 'slv', 'sme', 'sna',
                  'spa', 'sqi', 'srd', 'srn', 'stq', 'swa', 'swe', 'szl', 'tet', 'ton', 'tsn', 'vec', 'vep', 
                  'vls', 'vol', 'vro', 'war', 'wln', 'xho', 'yor', 'zea'],
    'cyrillic' : ['ava', 'bak', 'be-tarask', 'bel', 'bul', 'bxr', 'che', 'chv', 'crh', 'kaa', 'koi', 'krc', 
                  'mdf', 'mhr', 'mkd', 'mrj', 'myv', 'oss', 'rue', 'rus', 'sah', 'tat', 'tyv', 'ukr'],
    'arabic'   : ['ara', 'arz', 'azb', 'ckb', 'fas', 'glk', 'lrc', 'mzn', 'pnb', 'urd']
}
