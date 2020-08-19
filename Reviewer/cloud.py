import re
import os
import json
import math
import numpy as np
from PIL import Image
import colorsys

from wordcloud import WordCloud, ImageColorGenerator
from Reviewer.utils import MovieNotFoundError


class WordCloudGenerator:
    def __init__(self, dir_path: str = ""):
        self.dir_path = dir_path

    def generate_image(self,
                       mask: str,
                       pixels: int,
                       movie: str = None,
                       word_type: str = None,
                       path: str = None,
                       save: bool = False) -> Image.Image:
        """

        Parameters
        ----------
        path : str, default = None
            Path to location of count or tfidf data
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
        if path:
            with open(path, "r") as f:
                word_vals = json.load(f)

                if movie:
                    try:
                        word_vals = word_vals[movie]
                    except:
                        print(word_vals.keys())
                else:
                    word_vals = word_vals[list(word_vals.keys())[0]]
        else:
            word_vals = self.load_disney_data(word_type, movie)

        freq = self.preprocess_data(word_vals)
        mask = self.load_mask(mask, pixels)
        image = self.generate_word_cloud(freq, mask)

        if save:
            self.save_image(image)

        return image

    def load_disney_data(self, tfidf_type: str, movie: str) -> (dict, dict):
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
            with open(f'{self.dir_path}data/disney_tfidf.json') as f:
                disney = json.load(f)
        else:
            with open(f'{self.dir_path}data/disney_tfidf_relative.json') as f:
                disney = json.load(f)

        if disney.get(movie):
            return disney[movie]

        else:
            movies = list(disney.keys())
            raise MovieNotFoundError(movie, movies)

    def preprocess_data(self, word_vals: dict) -> (dict, str):
        """ Preprocess data for word cloud generation"""
        words = [val[0] for val in word_vals]
        values = [val[1] for val in word_vals]
        freq = {word.upper(): value for word, value in zip(words, values)}
        
#         for _ in range(2):
#             for key in result.keys():
#                 result[key.upper()] = result.pop(key)
        return freq

    def generate_word_cloud(self, freq: dict, mask: np.ndarray = None) -> Image.Image:
        """ Generate word cloud """
        wc = WordCloud(background_color="white", mode="RGB", max_words=2000, mask=mask, min_font_size=1,
                       color_func=lambda *args, **kwargs: "black", font_path=f"{self.dir_path}data/fonts/staatliches.ttf")
        wc.generate_from_frequencies(freq)
        wc.recolor(color_func=BrightImageColorGenerator(mask))
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


class BrightImageColorGenerator(ImageColorGenerator):
    """
    Brighten the colors of the ImageColorGenerator since they come
    out rather bleak...

    """
    def __call__(self, word, font_size, font_path, position, orientation, **kwargs):
        color = super().__call__(word, font_size, font_path, position, orientation, **kwargs)
        color = tuple([int(re.sub("[^0-9]", "", x)) / 255 for x in color.split()])
        h, l, s = colorsys.rgb_to_hls(*color)
        color = colorsys.hls_to_rgb(h=h, l=min(1, l * .9), s=min(1, s * 1.2))
        return "rgb(%d, %d, %d)" % (color[0] * 255, color[1] * 255, color[2] * 255)
