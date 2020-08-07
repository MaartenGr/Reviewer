import argparse
from Disney.scraper import Scraper


def parse_arguments():
    """ Parse command line inputs """
    parser = argparse.ArgumentParser(description='Scraper')
    parser.add_argument('--path', help='Chromedriver path', default="drivers/chromedriver.exe")
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    sc = Scraper(dir_path=args.path)
    sc.scrape()


if __name__ == "__main__":
    main()
