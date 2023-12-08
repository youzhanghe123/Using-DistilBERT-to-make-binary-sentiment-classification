from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
import pickle, os

from nltk.corpus import opinion_lexicon
import nltk
nltk.download("opinion_lexicon")

def ngrams_count_features(texts=[""],
                          vectorizer=None,
                          ngram_range=(1,3), 
                          max_features=4000,
                          min_df=3,
                          max_df=0.7,
                          save_pth="./models"):
    
    model_name = f"count_vectorizer_ngram{ngram_range}_max_{max_features}_dfminmax_{min_df}_{max_df}.pickle"
    
    if vectorizer is None:
        if os.path.exists(f"{save_pth}/{model_name}"):
            print(f"Load a pre-trained vectorizer: {model_name}")
            vectorizer = pickle.load(open(f"{save_pth}/{model_name}", "rb"))
        
        else:
            vectorizer = CountVectorizer(
                ngram_range=ngram_range, max_features=max_features, min_df=min_df, max_df=max_df
            )
            print("Fitting ngram counts over training set...")
            vectorizer.fit(texts)
            print(f"Fitted! Saving into {save_pth}") 
            pickle.dump(vectorizer, open(f"{save_pth}/{model_name}", "wb"))
        
    ngrams_count_feats = vectorizer.transform(texts)

    return vectorizer, ngrams_count_feats

def ngrams_tfidf_features(texts=[""],
                          vectorizer=None,
                          ngram_range=(1,3), 
                          max_features=4000,
                          min_df=3,
                          max_df=0.7,
                          save_pth="./models"):
    
    model_name = f"tfidf_vectorizer_ngram{ngram_range}_max_{max_features}_dfminmax_{min_df}_{max_df}.pickle"

    if vectorizer is None: # for training texts
        if os.path.exists(f"{save_pth}/{model_name}"):
            print(f"Load a pre-trained vectorizer: {model_name}")
            vectorizer = pickle.load(open(f"{save_pth}/{model_name}", "rb"))
        
        else:
            vectorizer = TfidfVectorizer(
                ngram_range=ngram_range, max_features=max_features, min_df=min_df, max_df=max_df
            )
            print("Fitting ngram tfidf over training set...")
            vectorizer.fit(texts)
            print(f"Fitted! Saving into {save_pth}") 
            pickle.dump(vectorizer, open(f"{save_pth}/{model_name}", "wb"))
    
    ngrams_tfidf_feats = vectorizer.transform(texts)

    return vectorizer, ngrams_tfidf_feats

def sentiment_lexicon_features(count_vectorizer, 
                               ngrams_count_feats, 
                               lexicon: dict = {}):

    counts_lexicon = lil_matrix((len(count_vectorizer.vocabulary_), len(lexicon)))
    num_lexicon_contains = 0
    for w in count_vectorizer.vocabulary_:
        if w in lexicon:
            counts_lexicon[count_vectorizer.vocabulary_[w], lexicon[w]] = 1
            num_lexicon_contains += 1
    print("Found {}/{} lexemes in training vocabulary".format(num_lexicon_contains, len(lexicon)))
    lexicon_vector = (ngrams_count_feats @ counts_lexicon).toarray()

    return lexicon_vector

def process_lexicon():
    
    lexicon = opinion_lexicon.words()
    lexicon = set(lexicon)
    
    print("Retrieved Sentiment Lexicon with length {}".format(len(lexicon)))
    
    lexicon = {word: i for i, word in enumerate(lexicon)}
    return lexicon

def craft_features(featset="tfidf",
                   text_splits={"train":[], "val":[], "test":[]},
                   feat_args=None,):

    all_featsets = ["tfidf", "tfidf+lexicon"]

    tfidf_vectorizer, X_train_tfidf = ngrams_tfidf_features(text_splits['train'], None,
                                                            feat_args.ngram_range,
                                                            feat_args.max_tfidf_features,
                                                            feat_args.min_df,
                                                            feat_args.max_df)
    X_val_tfidf = tfidf_vectorizer.transform(text_splits['val'])
    X_test_tfidf = tfidf_vectorizer.transform(text_splits['test'])

    if featset == all_featsets[0]:
        X_train, X_val, X_test = X_train_tfidf.toarray(), X_val_tfidf.toarray(), X_test_tfidf.toarray()

    elif featset == all_featsets[1]:
        count_vectorizer, X_train_count = ngrams_count_features(text_splits['train'],
                                                                vectorizer=None,
                                                                ngram_range=feat_args.ngram_range, 
                                                                max_features=None,
                                                                min_df=feat_args.min_df,
                                                                max_df=feat_args.max_df,)
        X_val_count = count_vectorizer.transform(text_splits['val'])
        X_test_count = count_vectorizer.transform(text_splits['test'])

        lexicon = process_lexicon()
        X_train_lexicon = sentiment_lexicon_features(count_vectorizer, X_train_count, lexicon)
        X_val_lexicon = sentiment_lexicon_features(count_vectorizer, X_val_count, lexicon)
        X_test_lexicon = sentiment_lexicon_features(count_vectorizer, X_test_count, lexicon)
        
        X_train, X_val, X_test = np.concatenate((X_train_tfidf.toarray(), X_train_lexicon), axis=1), \
                            np.concatenate((X_val_tfidf.toarray(), X_val_lexicon), axis=1), \
                        np.concatenate((X_test_tfidf.toarray(), X_test_lexicon), axis=1) 
    else:
        raise ValueError(f"featset should be either of {all_featsets}")

    return X_train, X_val, X_test

def vectorize_labels(label_splits={"train":[], "val":[], "test":[]}):

    y_train, y_val, y_test = np.array(label_splits['train']), \
                                np.array(label_splits['val']), \
                                    np.array(label_splits['test'])

    return y_train, y_val, y_test


class FEAT_ARG:

    def __init__(self, 
                 ngram_range, min_df, max_df,
                 max_tfidf_features
                 ) -> None:
        
        self.ngram_range = ngram_range
        self.max_tfidf_features = max_tfidf_features
        self.min_df = min_df
        self.max_df = max_df
