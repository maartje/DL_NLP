import unittest
import mock
import preprocess
import train
import predict
import evaluate
import config
import tests.mock_file_system as mfs


class TestPipeline(unittest.TestCase):

    @mock.patch('builtins.print')
    @mock.patch('preprocess.save_file', side_effect = mfs.mock_save_file)
    @mock.patch('preprocess.load_data', side_effect = mfs.mock_load_data)
    @mock.patch('torch.load', side_effect = mfs.mock_torch_load)
    @mock.patch('torch.save', side_effect = mfs.mock_torch_save)
    def test_pipeline(self, torch_save, torch_load, read_file, save_file, prnt = None):
        self.configure_for_testing()
        preprocess.main()
        train.main()
        predict.main()
        evaluate.main()

    def configure_for_testing(self):
        config.settings['rnn']['emb_size'] = 64
        config.settings['rnn']['hidden_size'] = 256
        config.settings['rnn']['drop_out'] = 0.3
        config.settings['rnn']['learning_rate'] = 1.
        config.settings['rnn']['epochs'] = 5

