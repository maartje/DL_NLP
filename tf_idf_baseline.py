from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
import config
from preprocess import load_data
import torch

def main():
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
    
    # fit a naive bayes classifier with tf-idf features
    pipeline = Pipeline([
        ('count_vectorizer',   CountVectorizer()),
        ('tfidf_transformer',  TfidfTransformer()), # optional
        ('classifier',         MultinomialNB())
    ])
    pipeline.fit(texts_train, targets_train)
    
    # TODO: we may consider doing something with all possible
    # completions for the last word
    
    # calculate accuracies pre position
    def accuracy_at_position(texts, targets, position):
        texts_cutoff = [t[0:position]for t in texts]
        predicted = pipeline.predict(texts_cutoff)
        results = (np.array(targets) == predicted)
        return results.sum()/len(results) 
    n = config.settings['max_seq_length']       
    accuracies_test = [
        accuracy_at_position(texts_test, targets_test, n) for n in range(n)
    ]    
    torch.save(accuracies_test, config.filepaths['tf_idf_test_accuracies'])
    

if __name__ == "__main__":
    main()
    

