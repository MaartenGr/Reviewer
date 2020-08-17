import os
import json
import math
import numpy as np
from PIL import Image
import colorsys

from wordcloud import WordCloud
from Reviewer.utils import MovieNotFoundError
from PIL import ImageFont


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
            
            


class ImageColorGenerator(object):
    """Color generator based on a color image.

    Generates colors based on an RGB image. A word will be colored using
    the mean color of the enclosing rectangle in the color image.

    After construction, the object acts as a callable that can be passed as
    color_func to the word cloud constructor or to the recolor method.

    Parameters
    ----------
    image : nd-array, shape (height, width, 3)
        Image to use to generate word colors. Alpha channels are ignored.
        This should be the same size as the canvas. for the wordcloud.
    default_color : tuple or None, default=None
        Fallback colour to use if the canvas is larger than the image,
        in the format (r, g, b). If None, raise ValueError instead.
    """
    # returns the average color of the image in that region
    def __init__(self, image, default_color=None):
        if image.ndim not in [2, 3]:
            raise ValueError("ImageColorGenerator needs an image with ndim 2 or"
                             " 3, got %d" % image.ndim)
        if image.ndim == 3 and image.shape[2] not in [3, 4]:
            raise ValueError("A color image needs to have 3 or 4 channels, got %d"
                             % image.shape[2])
        self.image = image
        self.default_color = default_color


    def __call__(self, word, font_size, font_path, position, orientation, **kwargs):
        """Generate a color for a given word using a fixed image."""
        # get the font to get the box size
        font = ImageFont.truetype(font_path, font_size)
        transposed_font = ImageFont.TransposedFont(font,
                                                   orientation=orientation)
        
        # get size of resulting text
        box_size = transposed_font.getsize(word)
        x = position[0]
        y = position[1]
        
        # cut out patch under word box
        patch = self.image[x:x + box_size[0], y:y + box_size[1]]
        if patch.ndim == 3:
            # drop alpha channel if any
            patch = patch[:, :, :3]
        if patch.ndim == 2:
            raise NotImplementedError("Gray-scale images TODO")
            
        # check if the text is within the bounds of the image
        reshape = patch.reshape(-1, 3)
        if not np.all(reshape.shape):
            if self.default_color is None:
                raise ValueError('ImageColorGenerator is smaller than the canvas')
            return "rgb(%d, %d, %d)" % tuple(self.default_color)
        color = np.mean(reshape, axis=0)
        h, l, s = colorsys.rgb_to_hls(*color/255)
        color = colorsys.hls_to_rgb(h=h,l=min(1, l*.9),s=min(1, s*1.2))
        
        return "rgb(%d, %d, %d)" % (color[0]*255, color[1]*255, color[2]*255)
#         return "rgb(%d, %d, %d)" % tuple(color)
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


