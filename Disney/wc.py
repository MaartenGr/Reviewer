import os
import json
import math
import numpy as np
from PIL import Image

from wordcloud import WordCloud, ImageColorGenerator
from Disney.utils import MovieNotFoundError


class WordCloudGenerator:
    def __init__(self, dir_path: str = ""):
        self.dir_path = dir_path

    def generate_image(self, movie: str, word_type: str, mask: str, pixels: int, save: bool = False) -> Image.Image:
        """

        Parameters
        ----------
        movie : str
            The name of the movie
        word_type : str
            Either "TF-IDF" or "TF-IDF-Relative"
        mask : str
            Name of the mask, for example, "coco.jpg"
        pixels : int
            Minimum number of pixels
        save : bool
            Whether to save the resulting image

        """
        word_vals, reviews = self.load_data(word_type, movie)
        freq, text = self.preprocess_data(word_vals, reviews)
        mask = self.load_mask(mask, pixels)
        image = self.generate_word_cloud(freq, mask)

        if save:
            self.save_image(image)

        return image

    def load_data(self, tfidf_type: str, movie: str) -> (dict, dict):
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
            with open(f'{self.dir_path}data/pixar_tfidf.json') as f:
                pixar = json.load(f)

            with open(f'{self.dir_path}data/disney_tfidf.json') as f:
                disney = json.load(f)
        else:
            with open(f'{self.dir_path}data/pixar_tfidf_relative.json') as f:
                pixar = json.load(f)

            with open(f'{self.dir_path}data/disney_tfidf_relative.json') as f:
                disney = json.load(f)

        if pixar.get(movie):
            with open(f'{self.dir_path}data/pixar_reviews.json') as f:
                reviews = json.load(f)
            return pixar[movie], reviews[movie]

        elif disney.get(movie):
            with open(f'{self.dir_path}data/disney_reviews.json') as f:
                reviews = json.load(f)
            return disney[movie], reviews[movie]

        else:
            movies = list(disney.keys()) + list(pixar.keys())
            raise MovieNotFoundError(movie, movies)

    def preprocess_data(self, word_vals: dict, reviews: dict) -> (dict, str):
        """ Preprocess data for word cloud generation"""
        words = [val[0] for val in word_vals]
        values = [val[1] for val in word_vals]
        freq = {word: value for word, value in zip(words, values)}
        text = " ".join([review[1] for review in reviews])
        return freq, text

    def generate_word_cloud(self, freq: dict, mask: np.ndarray = None) -> Image.Image:
        """ Generate word cloud """
        wc = WordCloud(background_color="white", mode="RGBA", max_words=2000, mask=mask, min_font_size=1,
                       color_func=lambda *args, **kwargs: "black")
        wc.generate_from_frequencies(freq)
        wc.recolor(color_func=ImageColorGenerator(mask))
        image = wc.to_image()
        return image

    def load_mask(self, url: str, min_pixels: int = 1500) -> np.ndarray:
        """ Open mask and resize it if it is too small. Should be at least 1000 x 1000 pixels """
        url = f"{self.dir_path}images/masks/" + url
        mask = Image.open(url)
        mask = mask.convert("RGB")
        if mask.size[0] < min_pixels or mask.size[1] < min_pixels:

            multiplier = math.ceil(min_pixels / min([mask.size[0], mask.size[0]]))
            mask = mask.resize((mask.size[0] * multiplier, mask.size[1] * multiplier), Image.ANTIALIAS)

        return np.array(mask)

    def save_image(self, image: Image.Image) -> None:
        """ Save output image

        Get all output images that were previously generated, extract their highest value
        and save the new image with the newest highest value. For example, if images ["result.png", "result1.png"]
        were to exist, then the name of the new image would be "result2.png".

        """

        # Find the image with the highest value in their name
        saved_images = [image for image in os.listdir(f"{self.dir_path}images/results") if "result" in image]
        highest_saved_image = 0
        for saved_image in saved_images:
            number = ''.join(i for i in saved_image if i.isdigit())

            if number:
                number = int(number)
                if number > highest_saved_image:
                    highest_saved_image = number

        # Save image
        if highest_saved_image == 0:
            image.save(f"{self.dir_path}images/results/result.png")
        else:
            image.save(f"{self.dir_path}images/results/result_{highest_saved_image+1}.png")

#
# def parse_arguments() -> argparse.Namespace:
#     """ Parse command line inputs """
#     masks = tuple(os.listdir("masks"))
#
#     parser = argparse.ArgumentParser(description='Scraper')
#     parser.add_argument('--movie', help='Movie', default="Frozen")
#     parser.add_argument('--type', help='Type of top words', choices=("tfidf", "relative"), default="TFIDF")
#     parser.add_argument('--mask', help='Mask url', choices=masks, default="coco.jpg")
#     parser.add_argument('--pixels', help='Minimum number of pixels', default=500, type=int)
#     args = parser.parse_args()
#
#     if args.type == "tfidf":
#         args.type = "TF-IDF"
#     else:
#         args.type = "TF-IDF-Relative"
#     return args
#
#
# def main():
#     args = parse_arguments()
#     word_vals, reviews = load_data(args.type, args.movie)
#     freq, text = preprocess_data(word_vals, reviews)
#     mask = load_mask(args.mask, args.pixels)
#     image = generate_word_cloud(freq, mask)
#     save_image(image)


# if __name__ == "__main__":
#     main()


