from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
import config
from preprocess import load_data
import torch
from src.preprocess.tokenizer import CharacterTokenizer, WordTokenizer

def cut_text_fragments(texts, max_length, tokenizer):
    if not config.settings['use_all_fragments']:
        texts = [tokenizer.get_prefix_fragment(s, max_length) for s in texts]
    return texts

def main():
    torch.save(
        {
            'word' : evaluate_naive_bayes(WordTokenizer(), 'word'),
            'char_word_features' : evaluate_naive_bayes(CharacterTokenizer(), 'word'),
            'char_char_features' : evaluate_naive_bayes(CharacterTokenizer(), 'char')
        },
        config.filepaths['naive_bayes_accuracies']
    )

def evaluate_naive_bayes(tokenizer, features):
    model_name = config.settings['model_name']
    # load test and train data
    lang_filter = config.language_filters[config.settings['language_filter']]
    texts_train, targets_train = load_data(
        config.filepaths['texts_train'], 
        config.filepaths['labels_train'], 
        lang_filter)
    texts_test, targets_test = load_data(
        config.filepaths['texts_test'], 
        config.filepaths['labels_test'], 
        lang_filter)

    texts_train = cut_text_fragments(
        texts_train, 
        config.settings[model_name]['max_seq_length'], 
        tokenizer)

    # fit a naive bayes classifier with tf-idf features
    pipeline = Pipeline([
        ('count_vectorizer',   CountVectorizer(analyzer = features)), # analyzer = 'word' or 'char'
        #('tfidf_transformer',  TfidfTransformer()), # counts seems more appropriate
        ('classifier',         MultinomialNB())
    ])
    pipeline.fit(texts_train, targets_train)
    
    # calculate accuracies pre position
    def accuracy_at_position(texts, targets, position):
        texts_cutoff = [tokenizer.get_prefix_fragment(s, position) for s in texts]
        predicted = pipeline.predict(texts_cutoff)
        results = (np.array(targets) == predicted)
        return results.sum()/len(results) 
    n = config.settings[model_name]['max_seq_length']       
    accuracies_test = [
        accuracy_at_position(texts_test, targets_test, n) for n in range(n)
    ]    
    return accuracies_test
    

if __name__ == "__main__":
    main()
    

