import re
import json

from tqdm import tqdm
from typing import List, Tuple

from flair.data import Sentence
from flair.models import SequenceTagger, TextClassifier
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')


class Character:
    """
    Extract characters from reviews and the sentiment corresponding
    to the sentences from which the characters were extracted. This way,
    you can analyze how positive/negative people are about certain characters.

    NOTE: This works better for actual people than the characters from movies
    or animations. Often, reviewers describe what happens in movies to certain
    characters which could be negative of nature, while it was not intended as
    a negative opinion of that character. Thus, it combines both descriptive events
    as well as opinions. However, since the actors themselves are not played in the movie,
    the sentences in which actors appear are typically of an opinionated nature.

    Parameters:
    -----------
    fast : bool, default = True
        Whether to use a cpu-based BERT classifier and tagger

    dir_path : str
        The path of the cwd, keep empty if there are no
        files to be saved in a parent dir

    """
    def __init__(self, fast=False, dir_path: str = ""):
        if fast:
            self.tagger = SequenceTagger.load('ner-fast')
            self.classifier = TextClassifier.load('sentiment-fast')
        else:
            self.tagger = SequenceTagger.load('ner')
            self.classifier = TextClassifier.load('sentiment')

        self.dir_path = dir_path
        self.reviews = None
        self.titles = None

    def predict_single_movie(self, name: str, reviews: List[str]) -> List[Tuple[str, int, str]]:
        """ Create predictions for a single movie


        Parameters
        ----------
        name : str
            Name of the movie

        reviews : list of str
            A list of reviews


        Returns
        -------
        results : list of tuples
            A list where each item is a tuple of the text, score and value if it is
            classified as "PER"

        """
        new_docs = [sent_tokenize(doc) for doc in reviews]
        new_docs = [x for sublist in new_docs for x in sublist]
        new_docs = [Sentence(x) for x in new_docs]

        self.tagger.predict(new_docs, verbose=False)
        self.classifier.predict(new_docs, verbose=False)

        results = []
        for sentence in new_docs:
            for token in sentence.get_spans('ner'):
                if token.tag == "PER":
                    results.append((token.text, sentence.get_labels()[0].score, sentence.get_labels()[0].value))

        with open(f'{self.dir_path}{name}.json', 'w') as f:
            json.dump(results, f)

        return results

    def load_reviews(self, path):
        """ Load reviews and the corresponding titles

        Parameters
        ----------
        path : str
            The path of the review location.
            E.g. : disney_reviews.json


        """
        with open(f'{self.dir_path}{path}') as f:
            self.reviews = json.load(f)

        self.titles = [(title, re.sub('[^a-zA-Z]+', '', title).lower()) for title in list(self.reviews.keys())]

    def predict(self, path: str, prefix: str) -> dict:
        """ For each movie in path


        Parameters
        ----------
        path : str
            The path of the review location.
            E.g. : disney_reviews.json

        prefix : str
            Prefix for the file you want saved

        Returns
        -------
        to_save : dict
            Title (key) and results (value) for each movie
        """
        self.load_reviews(path)

        # Generate predictions
        results = {title: None for title in self.titles}
        for title, name in tqdm(self.titles):
            results[title] = self.predict_single_movie(name, self.reviews[title])

        # Save results - make sure correct format is used
        to_save = {title: results[title] for title, _ in self.titles}

        with open(f'{self.dir_path}{prefix}_names.json', 'w') as f:
            json.dump(to_save, f)

        return to_save

# char = Character()
# results = char.predict("reviews.json", prefix="lotr")
