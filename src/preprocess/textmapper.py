from src.preprocess.vocabulary import Vocabulary
from itertools import chain

class TextMapper:

    def __init__(self, tokenizer, vocab = None):
        self.vocab = vocab
        self.UNKNOWN = "UNKNOWN"
        self.PAD = "PAD"
        self.tokenizer = tokenizer
    
    def build_vocabulary(self, sentences, min_occurence):
        self.vocab = Vocabulary()
        sentences_split = (
            self.tokenizer.sentence2tokens(
                self.tokenizer.preprocess(sentence)
            ) for sentence in sentences
        )
        words = chain.from_iterable(sentences_split)
        self.vocab.build(words, [self.PAD, self.UNKNOWN], min_occurence)

    def PAD_index(self):
        return self.vocab.word2index[self.PAD]

    def UNKNOWN_index(self):
        return self.vocab.word2index[self.UNKNOWN]

    def sentence2indices(self, sentence):
        return self.tokens2indices(self.tokenizer.sentence2tokens(sentence))

    def indices2sentence(self, indices):
        return self.tokenizer.tokens2sentence(self.indices2tokens(indices, True))

    def tokens2indices(self, tokens):
        return [self.token2index(t) for t in tokens]

    def token2index(self, t):
        return self.vocab.word2index.get(t, self.vocab.word2index[self.UNKNOWN])

    def remove_predefined_indices(self, indices):
        predefined = [
           self.token2index(self.UNKNOWN),
           self.token2index(self.PAD),
        ]
        return [
            i for i in indices if not i in predefined # remove UNKNOWN, PAD
        ] 
    
    def indices2tokens(self, indices, remove_predefined_tokens):
        if remove_predefined_tokens:
            indices = self.remove_predefined_indices(indices)
        return [self.index2token(i) for i in indices] 

    def index2token(self, i):
        return self.vocab.index2word[i]
        


