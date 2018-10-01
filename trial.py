from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
import config
from preprocess import load_data
import torch

def main():
   
    lang_filter = config.language_filters[config.settings['language_filter']]
    texts_train, targets_train = load_data(
        config.filepaths['texts_train'], 
        config.filepaths['labels_train'], 
        lang_filter)
    texts_test, targets_test = load_data(
        config.filepaths['texts_test'], 
        config.filepaths['labels_test'], 
        lang_filter)

    position = 2

    #texts_cutoff = [t[0:position]for t in texts_test]
    texts_cutoff =  [' '.join(t.split(" ")[0:position]) for t in texts_test]

    print(texts_cutoff[0:5])



if __name__ == "__main__":
    main()
    
    ' '.join

