"""
Basic scraping logic

Example:
python scrape.py --chrome "drivers/chromedriver"

"""

import argparse
from Disney.scraper import Scraper
from Disney.tfidf import TFIDF


def parse_arguments():
    """ Parse command line inputs """
    parser = argparse.ArgumentParser(description='Scraper')
    parser.add_argument('--chrome', help='Chromedriver path', default="drivers/chromedriver.exe")
    parser.add_argument('--path', help='Dir path', default="")
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    sc = Scraper(dir_path=args.path, chrome_path=args.chrome)
    sc.scrape_disney()

    tf = TFIDF(dir_path=args.path)
    tf.generate_disney()


if __name__ == "__main__":
    main()
