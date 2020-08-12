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


class Names:
    def __init__(self, dir_path: str = ""):
        self.dir_path = dir_path

        # NER tagger and sentiment classifier
        self.tagger = SequenceTagger.load('ner-fast')  # ner-fast for cpu-based predictions
        self.classifier = TextClassifier.load('sentiment-fast')

    def run(self, reviews_path: str, movie: str):
        reviews = self.load_reviews(reviews_path, movie)
        names = self.extract_names(reviews, save=movie)

    def load_reviews(self, reviews_path: str, movie: str) -> list:
        with open(f'{self.dir_path}data/{reviews_path}') as f:
            reviews = json.load(f)
        return reviews[movie]

    def extract_names(self, docs: list, save: str = False):
        """ Extract sentiment """

        results = []

        for doc in tqdm(docs):

            # split doc in sentences
            sentences = sent_tokenize(doc)
            # sentences = [e + "." for e in doc.split(".") if e]

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

