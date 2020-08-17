import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from unidecode import unidecode
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.models import TextClassifier
from rich import print
from nltk.tokenize import sent_tokenize

import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt

import nltk
nltk.download('punkt')

%matplotlib inline

def extract_names(docs, save=False):
    results = []
    
    for doc in docs:
        
        # split doc in sentences
        sentences = sent_tokenize(doc)
#         sentences = [e + "." for e in doc.split(".") if e]
        
        for sent in sentences:
            
            sentence = Sentence(sent, use_tokenizer=True)
            tagger.predict(sentence)

            for entity in sentence.get_spans('ner'):
                
                # Extract person and predict sentiment if person is in sentence
                if entity.tag == "PER":
                    classifier.predict(sentence)
                    score = sentence.get_labels()[0].score
                    value = sentence.get_labels()[0].value
                    name = entity.text
                    results.append((name, score, value))

    if save:
        with open(f'../data/{save}_names.json', 'w') as f:
            json.dump(results, f)
                
    return results


def extract_df(name_list):
    df = pd.DataFrame(name_list, columns=["Word", "Prob", "Sentiment"])
    df.Sentiment = df.Sentiment.map({"POSITIVE": 1, "NEGATIVE": -1})
    df.Word = df.Word.astype(str)
    df = preprocess_counts(df)
    return df


def preprocess_counts(df):
    """ Preprocess the data
    
    * Combine names such as Miguel and Miguel's by removing everything next to the special character 
    
    """
    # Remove accents
    df.Word = [unidecode(word) for word in df.Word]
    
    # Combine names such as Miguel and Miguel's by removing everything next to the special character
#     df.Word = df.Word.str.replace('[^A-z]',' ').str.strip().str.split(' ').str[0]  
    df.Word = df.Word.str.replace('[^A-z]',' ').str.strip()
    df = df.groupby("Word").agg({"Sentiment": [np.mean, np.count_nonzero]})
    df.columns = df.columns.droplevel()
    df = df.reset_index()
    df.columns = ["Word", "Sentiment", "Count"]
    df = df.sort_values("Count", ascending=False)
    
    # Remove names less than 3 characters
    df = df.loc[df.Word.str.len() >= 3, :]
    
    # Remove the "name" Disney
    df = df.loc[df.Word != "Disney", :]
    
    return df

def predict(title, name, reviews):
    print(title)
    new_docs = [sent_tokenize(doc) for doc in reviews]
    new_docs = [x for sublist in new_docs for x in sublist]
    total_length = sum([len(x) for x in new_docs])
    new_docs = [Sentence(x) for x in new_docs]
    
    tagger.predict(new_docs, verbose=True)
    classifier.predict(new_docs, verbose=True)

    results = []
    for sentence in new_docs:
        for token in sentence.get_spans('ner'):
            if token.tag == "PER":
                results.append((token.text, sentence.get_labels()[0].score, sentence.get_labels()[0].value))
                
    with open(f'../data/{name}.json', 'w') as f:
        json.dump(results, f)
    return results