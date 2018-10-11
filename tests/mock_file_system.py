mocked_file_storage = {
    'data/wili-2018/x_train.txt' : [
        'Dit is een konijn',
        'It is so nice to meet you',
        'Het is een mooie dag',
        'Hoe gaat het met je?',
        'I feel very happy',
        'Did you see what happened?'
    ],
    'data/wili-2018/y_train.txt' : [
        'nld',
        'eng',
        'nld',
        'nld',
        'eng',
        'eng'
    ],
    'data/wili-2018/x_test.txt' : [
        'Er is een kat in de boom',
        'Het is slecht weer vandaag',
        'Why are you crying?',
        'What is wrong whith you?'
    ],
    'data/wili-2018/y_test.txt' : [
        'nld',
        'nld',
        'eng',
        'eng'
    ]
}

def mock_torch_load(fpath, device=None):
    return mocked_file_storage[fpath]

def mock_torch_save(data, fpath):
    mocked_file_storage[fpath] = data

def mock_save_file(data, fpath):
    mocked_file_storage[fpath] = data

def mock_load_data(fpath_x, fpath_y, lang_filter):
    return mocked_file_storage[fpath_x], mocked_file_storage[fpath_y]