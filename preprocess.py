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
    # read targets from files
    targets_train = []
    targets_test = []
    with open('data/wili-2018/y_train.txt') as f1:
        for target in f1:
            targets_train.append(target)
    with open('data/wili-2018/y_test.txt') as f2:
        for target in f2:
            targets_train.append(target)

    # create dictionaries: 'label -> index' and 'index -> label',
    # i.e. ( { 0 -> 'eng', ...} , {'eng' -> 0, ...})
    label_to_index = {}
    all_labels = targets_train + targets_test
    index = 0
    for label in all_labels:
        if label in label_to_index: continue
        label_to_index[label] = index
        index += 1

    index_to_label = {index: label for label, index in label_to_index.items()}

    # convert targets_train and targets_test to list of indices
    targets_train_indices = [label_to_index[target] for target in targets_train]
    targets_test_indices = [label_to_index[target] for target in targets_test]

    # Store in 'data/preprocess':

    # TODO: store dictionairies: label2index and index2label

    with open('data/preprocess/targets_train_indices.pickle', 'wb') as h3:
        pickle.dump(targets_train_indices[0:10], h3, protocol=pickle.HIGHEST_PROTOCOL)

    with open('data/preprocess/targets_test_indices.pickle', 'wb') as h4:
        pickle.dump(targets_test_indices[0:10], h4, protocol=pickle.HIGHEST_PROTOCOL)

def preprocess_texts():
    sentences_train = []
    with open('data/wili-2018/x_train.txt') as f3:
        for sentence in f3:
            sentences_train.append(sentence)

    sentences_test = []
    with open('data/wili-2018/x_test.txt') as f4:
        for sentence in f4:
            sentences_test.append(sentence)

    min_occurrence = config.settings['min_occurrence']
    fpath_vocab = config.filepaths['vocab']
    fpath_vectors_train = config.filepaths['vectors_train']
    fpath_vectors_test = config.filepaths['vectors_test']

    # build vocab from training data
    build_vocabulary(sentences_train[0:10], min_occurrence, fpath_vocab)

    # build sentence vectors for train, validation and test sets
    build_vectors(sentences_train[0:10], fpath_vocab, fpath_vectors_train)
    build_vectors(sentences_test[0:10], fpath_vocab, fpath_vectors_test)


def main():
    preprocess_texts()
    preprocess_labels()

if __name__ == "__main__":
    main()