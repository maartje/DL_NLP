import unittest
import mock
import preprocess
import train

mocked_file_storage = {
}

def mock_torch_load(fpath):
    return mocked_file_storage[fpath]

def mock_torch_save(data, fpath):
    mocked_file_storage[fpath] = data

class TestPipeline(unittest.TestCase):

    @mock.patch('torch.load', side_effect = mock_torch_load)
    @mock.patch('torch.save', side_effect = mock_torch_save)
    def test_pipeline(self, torch_save, torch_load):
        preprocess.main()
        #train.main()

