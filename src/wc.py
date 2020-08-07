import argparse
import json
from utils import MovieNotFoundError
from rich import print
from PIL import Image
import numpy as np
import math
from wordcloud import WordCloud, ImageColorGenerator


def load_data(tfidf_type: str, movie: str) -> (dict, dict):
    """ Load all TF-IDF data for both Disney and Pixar

    Parameters:
    ------------
    tfidf_type: str
        The TF-IDF method to use, either regular or relative.
        Options:
            TF-IDF
            TF-IDF-Relative

    movie : str
        The movie to be used

    Returns:
    --------
    pixar, disney : (dict, dict)
        Each dict contains the top n word, value (key, val) pairs regarding
        the importance of each word.
    """

    if tfidf_type == "TF-IDF":
        with open('../data/pixar_tfidf.json') as f:
            pixar = json.load(f)

        with open('../data/disney_tfidf.json') as f:
            disney = json.load(f)
    else:
        with open('../data/pixar_tfidf_relative.json') as f:
            pixar = json.load(f)

        with open('../data/disney_tfidf_relative.json') as f:
            disney = json.load(f)

    if pixar.get(movie):
        with open('../data/pixar_reviews.json') as f:
            reviews = json.load(f)
        return pixar[movie], reviews[movie]

    elif disney.get(movie):
        with open('../data/disney_reviews.json') as f:
            reviews = json.load(f)
        return disney[movie], reviews[movie]

    else:
        movies = list(disney.keys()) + list(pixar.keys())
        raise MovieNotFoundError(movie, movies)


def preprocess_data(word_vals: dict, reviews: dict) -> (dict, str):
    """ Preprocess data for word cloud generation"""
    words = [val[0] for val in word_vals]
    values = [val[1] for val in word_vals]
    freq = {word: value for word, value in zip(words, values)}
    text = " ".join([review[1] for review in reviews])
    return freq, text


def generate_word_cloud(freq: dict, mask: np.ndarray = None):
    """ Generate word cloud """
    wc = WordCloud(background_color="white", mode="RGBA", max_words=2000, mask=mask, min_font_size=1,
                   color_func=lambda *args, **kwargs: "black")
    wc.generate_from_frequencies(freq)
    wc.recolor(color_func=ImageColorGenerator(mask))
    image = wc.to_image()
    return image


def load_mask(url: str) -> np.ndarray:
    """ Open mask and resize it if it is too small. Should be at least 1000 x 1000 pixels """
    url = "../images/" + url
    mask = Image.open(url)

    if mask.size[0] < 1000 and mask.size[1] < 1000:
        multiplier = math.ceil(1000 / mask.size[0])
        mask = mask.resize((mask.size[0] * multiplier, mask.size[1] * multiplier))

    return np.array(mask)


def parse_arguments() -> argparse.Namespace:
    """ Parse command line inputs """
    parser = argparse.ArgumentParser(description='Scraper')
    parser.add_argument('--movie', help='Movie', default="Frozen")
    parser.add_argument('--type', help='Type of top words', choices=("tfidf", "relative"), default="TFIDF")
    args = parser.parse_args()

    if args.type == "tfidf":
        args.type = "TF-IDF"
    else:
        args.type = "TF-IDF-Relative"
    return args


def main():
    args = parse_arguments()
    word_vals, reviews = load_data(args.type, args.movie)
    freq, text = preprocess_data(word_vals, reviews)
    mask = load_mask("coco_mask2")
    image = generate_word_cloud(freq, mask)


if __name__ == "__main__":
    main()


