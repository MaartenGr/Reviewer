# Data handling
import json
import numpy as np
import pandas as pd

# NLP
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class TFIDF:
    def __init__(self, dir_path: str = ""):
        self.dir_path = dir_path
        
    def generate(self, review_path: str, save_prefix: str):
            
        with open(review_path) as f:
            movie_reviews = json.load(f)
            
        for reviews in [(movie_reviews, save_prefix)]:
            titles, documents, m = self.prepare_data(reviews[0])
            tf_idf, count = self.c_tf_idf(documents, m, ngram_range=(1, 1))
            self.extract_top_n_tfidf(tf_idf, count, titles, n=2000, save=reviews[1])
            self.extract_top_n_relative_importance(tf_idf, count, titles, n=2000, save=reviews[1])

    def generate_disney(self):
        disney_reviews = self.load_disney_data()

        for reviews in [(disney_reviews, "disney")]:
            titles, documents, m = self.prepare_data(reviews[0])
            tf_idf, count = self.c_tf_idf(documents, m, ngram_range=(1, 1))
            self.extract_top_n_tfidf(tf_idf, count, titles, n=2000, save=reviews[1])
            self.extract_top_n_relative_importance(tf_idf, count, titles, n=2000, save=reviews[1])

    def c_tf_idf(self, documents, m, ngram_range=(1, 1)):
        """ Calculate Class-based TF-IDF

        The result is a single score for each word

        documents = list of documents where each entry contains a single string
        of each class. For example, let's say you have 200 documents per class and you have 2 classes.
        The documents is a list of two documents, where each document is a join of all 200 documents.

        m = total number of documents

        """

        count = CountVectorizer(ngram_range=ngram_range).fit(documents)
        t = count.transform(documents)
        t = np.array(t.todense()).T
        w = t.sum(axis=0)
        tf = np.divide(t + 1, w + 1)
        sum_tij = np.array(t.sum(axis=1)).T
        idf = np.log(np.divide(m, sum_tij)).reshape(-1, 1)
        tf_idf = np.multiply(tf, idf)

        return tf_idf, count

    def load_disney_data(self) -> (dict, dict):
        """ Load, for now, only Pixar reviews """
        with open(f'{self.dir_path}data/disney_reviews.json') as f:
            disney_reviews = json.load(f)
        return disney_reviews

    def prepare_data(self, reviews: dict) -> (list, list, int):
        """ Extract titles, documents and total number of documents (m)

        For each movie, all documents are joined such that each movie seemingly
        has a single, very long, review.

        """
        titles = list(reviews.keys())
        documents = [" ".join([doc for _, doc in reviews[title]]) for title in titles]
        m = sum([len(reviews[title]) for title in titles])

        return titles, documents, m

    def extract_top_n_tfidf(self, tf_idf, count, titles, n: int = 200, save: str = False):
        """ Extract the top n words for each movie based on their tf-idf score """
        result = pd.DataFrame(tf_idf, index=count.get_feature_names(), columns=titles)

        top_n_words = {movie: None for movie in titles}
        for movie in titles:
            words = result[[movie]].sort_values(movie, ascending=False).index[:n]
            values = result[[movie]].sort_values(movie, ascending=False).values[:n].flatten()
            top_n_words[movie] = [(word, value) for word, value in zip(words, values)]

        if save:
            with open(f'{self.dir_path}data/{save}_tfidf.json', 'w') as f:
                json.dump(top_n_words, f)

    def extract_top_n_relative_importance(self, tf_idf, count, titles, n: int = 200, save: str = False):
        """ Extract the top n words for each movie based on their relative tf-idf score """
        result = pd.DataFrame(tf_idf, index=count.get_feature_names(), columns=titles)

        top_n_words = {movie: None for movie in titles}
        for movie in titles:
            result["Importance"] = result[movie].values / result.drop(movie, 1).reset_index(drop=True).sum(axis=1).values
            words = result[["Importance"]].sort_values("Importance", ascending=False).index[:n]
            values = result[["Importance"]].sort_values("Importance", ascending=False).values[:n].flatten()
            top_n_words[movie] = [(word, value) for word, value in zip(words, values)]

        if save:
            with open(f'{self.dir_path}data/{save}_tfidf_relative.json', 'w') as f:
                json.dump(top_n_words, f)
