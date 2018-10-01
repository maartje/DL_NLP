#TODO: at some point we may want to experiment with other kind of tokenizers,
# such as word or BPE tokens. We can extract a abstract parent class for this.

class CharacterTokenizer(object):

    def get_prefix_fragment(self, sentence, max_length):
        return sentence[:max_length]

    def get_all_fragments(self, sentence, max_length):
        positions = range(0, len(sentence), max_length)
        return [
            sentence[i:i + max_length] for i in positions if i+max_length < len(sentence)
        ]

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


