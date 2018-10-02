from src.preprocess.textmapper import TextMapper
from src.preprocess.tokenizer import CharacterTokenizer
from src.preprocess.tokenizer import WordTokenizer
from src.preprocess.process_labels import process_labels
import torch
import config
import pickle



def build_vocabulary(sentences_train, min_occurence, tokenizer, fpath_vocab):
    mapper = TextMapper(tokenizer)
    mapper.build_vocabulary(sentences_train, min_occurence)
    torch.save(mapper, fpath_vocab)

def build_vectors(sentences, fpath_vocab, fpath_vectors):
    mapper = torch.load(fpath_vocab)
    sentence_vectors = [
        mapper.sentence2indices(sentence) for sentence in sentences
    ]
    torch.save(sentence_vectors, fpath_vectors)

def preprocess_targets(targets_train, targets_test):
    # create dictionairies for target values and transform labels into indices
    (
        targets_train_indices, 
        targets_test_indices, 
        label2index,
        index2label
    ) = process_labels(targets_train, targets_test)

    # TODO: store dictionaries: label2index and index2label in 'data/preprocess'
    save_file(targets_train_indices, config.filepaths['targets_train'])
    save_file(targets_test_indices, config.filepaths['targets_test'])
    save_file((label2index, index2label), config.filepaths['targets_dictionairies'])

def save_file(data, fpath):
    torch.save(data, fpath)
    # with open(fpath, 'wb') as f:
    #     pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_data(fpath_x, fpath_y, lang_filter):
    def read_data(fpath_x, fpath_y, lang_filter):
        with open(fpath_x, ) as x:
            with open(fpath_y) as y:
                for text, target in zip(x,y):
                    if target.strip() in lang_filter:
                        yield (text.strip(), target.strip())    
    data = list(read_data(fpath_x, fpath_y, lang_filter))
    data_unzipped = list(zip(*data))
    x_data = list(data_unzipped[0])
    y_data = list(data_unzipped[1])
    return x_data, y_data 

def preprocess_texts(sentences_train, sentences_test, tokenizer):
    min_occurrence = config.settings['min_occurrence']
    fpath_vocab = config.filepaths['vocab']
    fpath_vectors_train = config.filepaths['vectors_train']
    fpath_vectors_test = config.filepaths['vectors_test']

    # build vocab from training data
    build_vocabulary(sentences_train, min_occurrence, tokenizer, fpath_vocab)

    # build sentence vectors for train, validation and test sets
    build_vectors(sentences_train, fpath_vocab, fpath_vectors_train)
    build_vectors(sentences_test, fpath_vocab, fpath_vectors_test)

def split_in_fragments(texts, targets, tokenizer, max_length):
    new_texts = []
    new_targets = []
    for txt, trg in zip(texts, targets):
        fragments = tokenizer.get_all_fragments(txt, max_length)
        for frgm in fragments:
            new_texts.append(frgm)
            new_targets.append(trg)
    return new_texts, new_targets


def max_length_check(x_test, y_test, max_length):
    x_new = []
    y_new = []
    for i , s in enumerate(x_test):
        s = s.split(" ")
        if  len(s) >= max_length:
            x_new.append(' '.join(s))
            y_new.append(y_test[i])

    return x_new, y_new


def main():
    lang_filter_setting = config.settings['language_filter']
    lang_filter = config.language_filters[lang_filter_setting]
    model =  config.settings['model'] # char or word

    if model == 'char':
        tokenizer = CharacterTokenizer()
    else:
        tokenizer = WordTokenizer()

    x_train, y_train = load_data(
        config.filepaths['texts_train'], config.filepaths['labels_train'], lang_filter)
    x_test, y_test = load_data(
        config.filepaths['texts_test'], config.filepaths['labels_test'], lang_filter)
    max_length = config.settings['max_seq_length']


    #if model == 'word':
    #    x_test, y_test = max_length_check(x_test, y_test, max_length)


    if config.settings['use_all_fragments']:
        x_train, y_train = split_in_fragments(x_train, y_train, tokenizer, max_length)
    else:
        x_train = [tokenizer.get_prefix_fragment(s, max_length) for s in x_train]

    x_test = [tokenizer.get_prefix_fragment(s, max_length) for s in x_test] # we test on prefixes only
    preprocess_texts(x_train, x_test, tokenizer)
    preprocess_targets(y_train, y_test)

if __name__ == "__main__":
    main()