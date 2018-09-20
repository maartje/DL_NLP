from src.preprocess.textmapper import TextMapper
from src.preprocess.tokenizer import CharacterTokenizer
import torch
import config

def build_vocabulary(sentences_train, min_occurence, fpath_vocab):
    mapper = TextMapper(CharacterTokenizer())
    mapper.build_vocabulary(sentences_train, min_occurence)
    torch.save(mapper, fpath_vocab)

def build_vectors(sentences, fpath_vocab, fpath_vectors):
    mapper = torch.load(fpath_vocab)
    sentence_vectors = [
        mapper.sentence2indices(sentence) for sentence in sentences
    ]
    torch.save(sentence_vectors, fpath_vectors)


def main():
    # TODO: read from files
    sentences_train = ['hello', 'I am here!', 'this is interesting.'] 
    sentences_val = ['hello', 'I like being here!'] 
    sentences_test = ['hello world!', 'I am here!'] 
    
    min_occurrence = config.settings['min_occurrence']
    fpath_vocab = config.filepaths['vocab']
    fpath_vectors_train = config.filepaths['vectors_train']
    fpath_vectors_val = config.filepaths['vectors_val']
    fpath_vectors_test = config.filepaths['vectors_test']

    # build vocab from training data
    build_vocabulary(sentences_train, min_occurrence, fpath_vocab)

    # build sentence vectors for train, validation and test sets
    build_vectors(sentences_train, fpath_vocab, fpath_vectors_train)
    build_vectors(sentences_val, fpath_vocab, fpath_vectors_val)
    build_vectors(sentences_test, fpath_vocab, fpath_vectors_test)


if __name__ == "__main__":
    main()
