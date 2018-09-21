import unittest
import mock
import preprocess
import train
import config

mocked_file_storage = {
}

def mock_torch_load(fpath):
    return mocked_file_storage[fpath]

def mock_torch_save(data, fpath):
    mocked_file_storage[fpath] = data

def mock_save_file(data, fpath):
    mocked_file_storage[fpath] = data

def mock_read_file(fpath):
    if fpath == 'data/wili-2018/x_train.txt':
        return [
            'Dit is een konijn',
            'It is so nice to meet you',
            'Het is een mooie dag',
            'Hoe gaat het met je?',
            'I feel very happy',
            'Did you see what happened?'
        ]
    if fpath == 'data/wili-2018/y_train.txt':
        return [
            'nld',
            'eng',
            'nld',
            'nld',
            'eng',
            'eng'
        ]
    if fpath == 'data/wili-2018/x_test.txt':
        return [
            'Er is een kat in de boom',
            'Het is slecht weer vandaag',
            'Why are you crying?',
            'What is wrong whith you?'
        ]
    if fpath == 'data/wili-2018/y_test.txt':
        return [
            'nld',
            'nld',
            'eng',
            'eng'
        ]

class TestPipeline(unittest.TestCase):

    @mock.patch('preprocess.save_file', side_effect = mock_save_file)
    @mock.patch('preprocess.read_file', side_effect = mock_read_file)
    @mock.patch('torch.load', side_effect = mock_torch_load)
    @mock.patch('torch.save', side_effect = mock_torch_save)
    def test_pipeline(self, torch_save, torch_load, read_file, save_file):
        self.configure_for_testing()
        preprocess.main()
        train.main()

    def configure_for_testing(self):
        # config.settings['rnn'] = {}
        config.settings['rnn']['emb_size'] = 64
        config.settings['rnn']['hidden_size'] = 256
        config.settings['rnn']['drop_out'] = 0.3
        config.settings['rnn']['learning_rate'] = 1.
        config.settings['rnn']['epochs'] = 50

