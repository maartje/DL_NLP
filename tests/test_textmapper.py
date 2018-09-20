"""
Tests for the text mapper that maps words to indices and indices to words
"""

import unittest
from src.preprocess.textmapper import TextMapper
from src.preprocess.tokenizer import CharacterTokenizer

class TestTextMapper(unittest.TestCase):

    def setUp(self):
        sentences = ['Hello world!', 'Hello Foo.']
        self.mapper = TextMapper(CharacterTokenizer())
        self.mapper.build_vocabulary(sentences, 2)

    def test_index_special_tokens(self):
        self.assertEqual(self.mapper.PAD_index(), 0)
        self.assertEqual(self.mapper.UNKNOWN_index(), 1)

    def test_sentence2indices(self):
        sentence = 'helXlo'
        indices = self.mapper.sentence2indices(sentence)
                
        self.assertEqual(self.mapper.UNKNOWN_index(), indices[3])

    def test_indices2sentence(self):
        sentence = 'helXlo'
        indices = self.mapper.sentence2indices(sentence)
        sentence_out = self.mapper.indices2sentence(indices)
        sentence_expected = 'hello'
        
        self.assertEqual(sentence_expected, sentence_out)

if __name__ == '__main__':
    unittest.main()


