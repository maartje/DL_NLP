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
        
    def test_sentence2indices(self):
        sentence = 'hello!'
        indices = self.mapper.sentence2indices(sentence)
        SOS_index = self.mapper.vocab.word2index[self.mapper.SOS]
        EOS_index = self.mapper.EOS_index()
        UNKNOWN_index = self.mapper.vocab.word2index[self.mapper.UNKNOWN]
                
        self.assertEqual(SOS_index, indices[0])
        self.assertEqual(UNKNOWN_index, indices[-2])
        self.assertEqual(EOS_index, indices[-1])

    def test_indices2sentence(self):
        sentence = 'helXlo'
        indices = self.mapper.sentence2indices(sentence)
        sentence_out = self.mapper.indices2sentence(indices)
        sentence_expected = 'hello'
        
        self.assertEqual(sentence_expected, sentence_out)

if __name__ == '__main__':
    unittest.main()

