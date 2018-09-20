from src.preprocess.textmapper import TextMapper
from src.preprocess.tokenizer import CharacterTokenizer
import torch
import config
import pickle

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
    targets_train = ['nld', 'eng', 'frn', 'eng'] 
    targets_test = ['eng', 'nld']

    targets_train = []
    targets_test = []
    with open('../wili-2018/y_train.txt') as f1:
        for target in f1:
            targets_train.append(target)
    with open('../wili-2018/y_test.txt') as f2:
        for target in f2:
            targets_train.append(target)

    # TODO create dictionaries: 'label -> index' and 'index -> label'
    # - dictionaries, i.e. ( { 0 -> 'eng', ...} , {'eng' -> 0, ...})
    label_to_index = {}
    all_labels = targets_train + targets_test
    index = 0
    for label in all_labels:
        if label in label_to_index: continue
        label_to_index[label] = index
        index += 1

    index_to_label = {index: label for label, index in label_to_index.items()}

    # TODO convert targets_train and targets_test to list of indices
    # - list of indices for training targets, i.e. [3,16, ...]
    # - list of indices for test targets
    targets_train_indices = [label_to_index[target] for target in targets_train]
    targets_test_indices = [label_to_index[target] for target in targets_test]

    # TODO store in three different files in 'data/preprocess':
    with open('data/preprocess/targets_train.pickle', 'wb') as h1:
        pickle.dump(targets_train, h1, protocol=pickle.HIGHEST_PROTOCOL)

    with open('data/preprocess/targets_test.pickle', 'wb') as h2:
        pickle.dump(targets_test, h2, protocol=pickle.HIGHEST_PROTOCOL)

    with open('data/preprocess/targets_train_indices.pickle', 'wb') as h3:
        pickle.dump(targets_train_indices, h3, protocol=pickle.HIGHEST_PROTOCOL)

    with open('data/preprocess/targets_test_indices.pickle', 'wb') as h4:
        pickle.dump(targets_test_indices, h4, protocol=pickle.HIGHEST_PROTOCOL)

def preprocess_texts():
    # TODO: read from files
    sentences_train = ['Hallo wereld!', 'I am here!', 'je suis content.', 'I am happy.'] 
    sentences_test = ['hello world!', 'Ik loop de trap op'] 

    sentences_train = []
    with open('../wili-2018/x_train.txt') as f3:
        for sentence in f3:
            sentences_train.append(sentence)

    sentences_test = []
    with open('../wili-2018/x_test.txt') as f4:
        for sentence in f4:
            sentences_test.append(sentence)

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
