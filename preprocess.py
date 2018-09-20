from src.preprocess.textmapper import TextMapper
from src.preprocess.tokenizer import CharacterTokenizer
from src.preprocess.process_labels import process_labels
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

def preprocess_targets():
    # read targets from files
    targets_train = read_file(config.filepaths['labels_train'])
    targets_test = read_file(config.filepaths['labels_test'])


    # Store in 'data/preprocess':
    (
        targets_train_indices, 
        targets_test_indices, 
        label_to_index, 
        index_to_label
    ) = process_labels(targets_train, targets_test)

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
    preprocess_targets()

if __name__ == "__main__":
    main()