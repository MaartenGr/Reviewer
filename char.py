"""
Create Character Popularity Visualization

Full Example:
    python char.py --movie Frozen --extract True --fast True --prefix disney --rpath disney_reviews.json --actors False

Visualization only:
    python char.py --movie Frozen --prefix disney --rpath disney_reviews.json --npath disney_names.json --actors False

"""

import argparse
from Reviewer.names import Character


def parse_arguments() -> argparse.Namespace:
    """ Parse command line inputs """
    parser = argparse.ArgumentParser(description='Character')
    parser.add_argument('--movie', help='The name of the movie', required=True)
    parser.add_argument('--extract', help='Whether to extract names or use '
                                          'already extracted names from --npath', default=False)
    parser.add_argument('--fast', help='Whether to use cpu (True) or gpu (False) '
                                       'when extracting names from --rpath ', default=True)
    parser.add_argument('--prefix', help='Prefix for saving files', required=True)
    parser.add_argument('--rpath', help='Path to review data. E.g., disney_reviews.json. Note:'
                                        'This should be in the data folder', type=str, required=True)
    parser.add_argument('--npath', help='Path to names data. E.g., disney_names.json. Note:'
                                        'This should be in the data folder', type=str)
    parser.add_argument('--actors', help='Whether to only select names of two words which should '
                                         'represent people with the exception of characters with 2 names', default=False)
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    # Extract names + sentiment
    if args.extract:
        char = Character(dir_path="", fast=args.fast)
        char.predict(path="data/"+args.rpath, prefix=args.prefix)
        args.npath = args.prefix + "_names.json"

    # Visualize results
    char = Character(dir_path="", load_classifiers=False)
    char.preprocess_names_and_reviews("data/"+args.rpath, "data/"+args.npath)
    char.visualize_names(name=args.movie, people=args.actors, save=args.prefix)


if __name__ == "__main__":
    main()
