"""
Create Wordclouds

Movie Example:
    python scrape.py --path "data/some_movie_count.json" --mask some_mask.jpg --pixels 1200

Disney Example:
    python scrape.py --movie Coco --type tfidf --mask coco.jpg --pixels 1200

"""


import os
import argparse
from Reviewer.cloud import WordCloudGenerator


def parse_arguments() -> argparse.Namespace:
    """ Parse command line inputs """
    masks = tuple(os.listdir("images/masks"))

    parser = argparse.ArgumentParser(description='Scraper')
    parser.add_argument('--movie', help='Movie', default="Frozen")
    parser.add_argument('--type', help='Type of top words', choices=("tfidf", "relative"), default="TFIDF")
    parser.add_argument('--mask', help='Mask url', choices=masks, default="coco.jpg")
    parser.add_argument('--pixels', help='Minimum number of pixels', default=500, type=int)
    parser.add_argument('--path', help='Path to count or tfidf data', type=str)
    args = parser.parse_args()

    if args.type == "tfidf":
        args.type = "TF-IDF"
    else:
        args.type = "TF-IDF-Relative"
    return args


def main():
    args = parse_arguments()
    wc = WordCloudGenerator()
    wc.generate_image(movie=args.movie, word_type=args.type, mask=args.mask, pixels=args.pixels, save=True,
                      path=args.path)


if __name__ == "__main__":
    main()