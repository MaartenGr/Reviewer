"""
Create Wordclouds

Example:
python scrape.py --chrome "drivers/chromedriver"

"""


import os
import argparse
from Disney.cloud import WordCloudGenerator


def parse_arguments() -> argparse.Namespace:
    """ Parse command line inputs """
    masks = tuple(os.listdir("images/masks"))

    parser = argparse.ArgumentParser(description='Scraper')
    parser.add_argument('--movie', help='Movie', default="Frozen")
    parser.add_argument('--type', help='Type of top words', choices=("tfidf", "relative"), default="TFIDF")
    parser.add_argument('--mask', help='Mask url', choices=masks, default="coco.jpg")
    parser.add_argument('--pixels', help='Minimum number of pixels', default=500, type=int)
    args = parser.parse_args()

    if args.type == "tfidf":
        args.type = "TF-IDF"
    else:
        args.type = "TF-IDF-Relative"
    return args


def main():
    args = parse_arguments()
    wc = WordCloudGenerator()
    wc.generate_image(movie=args.movie, word_type=args.type, mask=args.mask, pixels=args.pixels, save=True)


if __name__ == "__main__":
    main()