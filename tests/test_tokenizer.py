"""
Tests for the text processor that tokenizes and preprocesses the text
"""

import unittest
from src.preprocess.tokenizer import CharacterTokenizer

class TestTokenizer(unittest.TestCase):
    
    def setUp(self):
        self.char_tokenizer = CharacterTokenizer()

    def test_preprocess(self):
        sentence = 'Hello world!  \n'
        sentence_expected = 'hello world!'
        sentence_preprocessed = self.char_tokenizer.preprocess(sentence)
        self.assertEqual(sentence_expected, sentence_preprocessed)

    def test_sentence2tokens(self):
        sentence = 'hello!'
        tokens = self.char_tokenizer.sentence2tokens(sentence)
        tokens_expected = ['h', 'e', 'l', 'l', 'o', '!']
        self.assertEqual(tokens_expected, tokens)

    def test_tokens2sentence(self):
        sentence = 'hello!'
        tokens = self.char_tokenizer.sentence2tokens(sentence)
        sentence_untokenized = self.char_tokenizer.tokens2sentence(tokens)
        self.assertEqual(sentence, sentence_untokenized)

if __name__ == '__main__':
    unittest.main()


