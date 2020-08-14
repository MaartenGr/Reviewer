"""
Basic scraping logic

Examples:

* To scrape a single movie:
    python scraper.py --prefix car --url https://www.imdb.com/title/tt1216475/reviews?ref_=tt_ov_rt --ngram 3
* To scrape all disney movies:
    python scraper.py --disney --ngram 3

"""
import json
import argparse
from Reviewer.scraper import Scraper
from Reviewer.tfidf import TFIDF


def parse_arguments():
    """ Parse command line inputs """
    parser = argparse.ArgumentParser(description='Scraper')
    parser.add_argument('--prefix', help='Prefix for saving files', default="")
    parser.add_argument('--path', help='Dir path', default="")
    parser.add_argument('--urls_path', help='Url path', default=False)
    parser.add_argument('--url', help='Url', default=False)
    parser.add_argument('--disney', dest='disney', action='store_true', help="Choose all disney movies")
    parser.add_argument('--ngram', help='Max ngram', default=2)

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    sc = Scraper(dir_path=args.path, prefix=args.prefix)

    # Extract url
    if args.disney:
        urls = sc.get_disney_urls()
    elif args.url:
        urls = [args.url]
    elif args.urls_path:
        with open(args.urls_path, "r") as f:
            urls = json.load(f)

    # Scrape data
    sc.scrape(urls)
    sc.parse_data()

    # Apply TF-IDF
    tf = TFIDF(dir_path=args.path)
    if args.disney:
        tf.generate_disney()
    else:
        tf.generate(review_path=f"{sc.dir_path}data/{args.prefix}reviews.json", save_prefix=args.prefix, max_ngram=args.ngram)


if __name__ == "__main__":
    main()
