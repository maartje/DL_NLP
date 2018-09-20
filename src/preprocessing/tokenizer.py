#TODO: at some point we may want to experiment with other kind of tokenizers,
# such as word or BPE tokens. We can extract a abstract parent class for this.

class CharacterTokenizer(object):

    def preprocess(self, sentence): 
        return sentence.lower().strip()

    def sentence2tokens(self, sentence):
        """
        Breaks a sentence in a list of characters that form the 
        tokens in the vocabulary.
        """
        
        return list(sentence)

    def tokens2sentence(self, tokens):
        """
        Builds a sentence from a list of characters. This function is the exact reverse
        of the function 'sentence2tokens'
        """

        return ''.join(tokens)


