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
    targets_train = read_file(config.filepaths['labels_train'])
    targets_test = read_file(config.filepaths['labels_test'])

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
    save_file(targets_train_indices, config.filepaths['targets_train'])
    save_file(targets_test_indices, config.filepaths['targets_test'])

def save_file(data, fpath):
    with open(fpath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def read_file(fpath):
    lines = []
    with open(fpath) as f1:
        for line in f1:
            lines.append(line)
    return lines[0:10] #TODO: temporarily we only read lines 0 to 10

def preprocess_texts():
    sentences_train = read_file(config.filepaths['texts_train'])
    sentences_test = read_file(config.filepaths['texts_test'])

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