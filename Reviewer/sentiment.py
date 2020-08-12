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

import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt


class Tagger:
    def __init__(self):
        self.tagger = SequenceTagger.load('ner-fast')
        self.classifier = TextClassifier.load('sentiment-fast')

    def extract_names(self, docs, save=False):
        results = []

        for doc in tqdm(docs):

            # split doc in sentences
            sentences = [e + "." for e in doc.split(".") if e]

            for sent in sentences:

                sentence = Sentence(sent, use_tokenizer=True)
                self.tagger.predict(sentence)

                for entity in sentence.get_spans('ner'):

                    # Extract person and predict sentiment if person is in sentence
                    if entity.tag == "PER":
                        self.classifier.predict(sentence)
                        score = sentence.get_labels()[0].score
                        value = sentence.get_labels()[0].value
                        name = entity.text
                        results.append((name, score, value))

        if save:
            with open(f'../data/{save}_names.json', 'w') as f:
                json.dump(results, f)

        return results

    def extract_df(self, name_list):
        df = pd.DataFrame(name_list, columns=["Word", "Prob", "Sentiment"])
        df.Sentiment = df.Sentiment.map({"POSITIVE": 1, "NEGATIVE": -1})
        df.Word = df.Word.astype(str)
        df = self.preprocess_counts(df)
        return df

    def preprocess_counts(self, df):
        """ Preprocess the data

        * Combine names such as Miguel and Miguel's by removing everything next to the special character

        """
        # Remove accents
        df.Word = [unidecode(word) for word in df.Word]

        # Combine names such as Miguel and Miguel's by removing everything next to the special character
        #     df.Word = df.Word.str.replace('[^A-z]',' ').str.strip().str.split(' ').str[0]
        df.Word = df.Word.str.replace('[^A-z]', ' ').str.strip()
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