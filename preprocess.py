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

def preprocess_labels():
    # TODO: read from files
    labels_train = ['nld', 'eng', 'frn', 'eng'] 
    labels_test = ['eng', 'nld'] 

    # TODO create dictionairies: 'label -> index' and 'index -> label'

    # TODO convert targets_train and targets_test to list of indices

    # TODO store in three different files in 'data/preprocess':
    # - dictionairies, i.e. ( { 0 -> 'eng', ...} , {'eng' -> 0, ...})
    # - list of indices for training targets, i.e. [3,16, ...]
    # - list of indices for test targetts

    targets_train = [1, 2, 3, 2] # TODO  
    targets_test = [2, 1] # TODO

    torch.save(targets_train, config.filepaths['targets_train'])
    torch.save(targets_test, config.filepaths['targets_test'])


def preprocess_texts():
    # TODO: read from files
    sentences_train = ['Hallo wereld!', 'I am here!', 'je suis content.', 'I am happy.'] 
    sentences_test = ['hello world!', 'Ik loop de trap op'] 
    
    min_occurrence = config.settings['min_occurrence']
    fpath_vocab = config.filepaths['vocab']
    fpath_vectors_train = config.filepaths['vectors_train']
    fpath_vectors_test = config.filepaths['vectors_test']

    # build vocab from training data
    build_vocabulary(sentences_train, min_occurrence, fpath_vocab)

    # build sentence vectors for train, validation and test sets
    build_vectors(sentences_train, fpath_vocab, fpath_vectors_train)
    build_vectors(sentences_test, fpath_vocab, fpath_vectors_test)


def main():
    preprocess_texts()
    preprocess_labels()

if __name__ == "__main__":
    main()
