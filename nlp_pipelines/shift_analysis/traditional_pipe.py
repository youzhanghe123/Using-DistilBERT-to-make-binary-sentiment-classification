from preprocess import clean_text
from features import craft_features, vectorize_labels, FEAT_ARG
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import argparse
import pandas as pd


def _load_data(pth, splits):

    splitted_texts = {split:[] for split in splits}
    splitted_labels = {split:[] for split in splits}

    for split in splits:

        df = pd.read_csv(f"{pth}/{split}.csv")
        if "cleaned_text" not in df.columns: 
            df["cleaned_text"] = df.loc[:, "text"].map(lambda t: clean_text(t))
            print(f"Cleaning text for {pth}/{split}.csv")
            df.to_csv(f"{pth}/{split}.csv", index=False)

        splitted_texts[split] = df.loc[:, "cleaned_text"].tolist()
        splitted_labels[split] = df.loc[:, "label"].tolist()

    return splitted_texts, splitted_labels

def _load_model(pth):

    model = LogisticRegression()
    return model


def pipe(model, splitted_texts, splitted_labels):
    FEATURESET = "tfidf"
    X_train, X_val, X_test = craft_features(featset=FEATURESET, text_splits=splitted_texts, feat_args=args)
    y_train, y_val, y_test = vectorize_labels(splitted_labels)


    
    