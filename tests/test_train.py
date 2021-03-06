import unittest
import mock
import preprocess
import train
import predict
import evaluate
import config
import tests.mock_file_system as mfs
import torch

class TestTrain(unittest.TestCase):

    @mock.patch('builtins.print')
    @mock.patch('preprocess.save_file', side_effect = mfs.mock_save_file)
    @mock.patch('preprocess.load_data', side_effect = mfs.mock_load_data)
    @mock.patch('torch.load', side_effect = mfs.mock_torch_load)
    @mock.patch('torch.save', side_effect = mfs.mock_torch_save)
    def test_train(self, torch_save, torch_load, read_file, save_file, prnt = None):
        self.configure_for_testing()
        preprocess.main()
        train.main()

        step_size = 20 # robustness: make sure train loss decreases after 20 epochs
        train_losses = mfs.mocked_file_storage[config.filepaths['epoch_metrics']]['train_losses']
        train_losses_step = train_losses[::step_size]
        is_decreasing = lambda l: all(l[i] > l[i+1] for i in range(len(l)-1))
        self.assertTrue(is_decreasing(train_losses_step))


    def configure_for_testing(self):
        config.settings['model'] = 'char'
        config.settings['model_name'] = 'rnn'
        config.settings['check_equal_seq_length'] = False
        config.settings['rnn']['hidden_size'] = 256
        config.settings['rnn']['drop_out'] = 0.3
        config.settings['rnn']['learning_rate'] = 0.05
        config.settings['rnn']['epochs'] = 61
        torch.manual_seed(7)


