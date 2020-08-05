import re
import time
import json
import requests
import argparse
import pandas as pd
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from bs4.element import Tag


def get_all_disney_titles() -> pd.DataFrame:
    """ Get all Disney titles and their release dates """
    url = 'https://en.wikipedia.org/wiki/List_of_Walt_Disney_Animation_Studios_films'
    df = pd.read_html(url, header=0)[1]
    df['Year'] = df.apply(lambda row: row['Release date'].split(",")[-1].strip(), 1)
    return df


def get_all_pixar_titles() -> pd.DataFrame:
    """ Get all Pixar titles and their release dates """
    url = "https://en.wikipedia.org/wiki/List_of_Pixar_films"
    df = pd.read_html(url, header=0)[0]
    df = df.loc[df.Film != "Released films", :]
    df = df.iloc[:22]
    df['Year'] = df.apply(lambda row: row['Release date'].split(",")[-1].strip(), 1)
    return df


def match_years(search_result: Tag, year: str) -> bool:
    """ Check if the year of a movie search matches (within 2 years) the year of the search result"""
    string = search_result.text
    year = int(year[-4:])

    # Extract year from string
    string_year = re.sub('[^0-9]', ' ', string)  # Keep numbers
    string_year = re.sub(' +', ' ', string_year).strip()  # Remove duplicate whitespaces
    string_year = int(string_year.split(" ")[-1])

    if abs(string_year - year) <= 2:
        return True

    return False


def scrape_imdb_urls(df: pd.DataFrame, save: str = None) -> dict:
    """ Scrape IMDB urls of all the movies """
    titles = list(df.Film.values)
    search_terms = (df.Film + "%20" + df.Year).values
    urls = [None for _ in range(len(search_terms))]

    # Search for the movie and extract the first result
    for index, search_term in tqdm(enumerate(search_terms), "Scraping IMDB urls"):

        # Get search result page
        search_url = f"https://www.imdb.com/find?q={search_term}&s=tt&ttype=ft&ref_=fn_ft"
        res = requests.get(search_url).text
        soup = BeautifulSoup(res, 'lxml')

        # Extract best search result
        for result in soup.find_all("td", class_="result_text"):
            if match_years(result, search_term):
                url = result.find_all("a", href=True)[0]["href"]
                url = f"https://www.imdb.com{url}reviews"
                urls[index] = url
                break

    # Need to manually add saludos amigos as imdb's search engine cannot find it
    urls = {title: url if url else "https://www.imdb.com/title/tt0036326/reviews" for url, title in zip(urls, titles)}

    # Also cannot find onward correctly...
    if "Onward" in urls:
        urls["Onward"] = "https://www.imdb.com/title/tt7146812/reviews"

    if save:
        with open(f'../data/{save}_urls.json', 'w') as f:
            json.dump(urls, f)

    return urls


def extract_reviews(soup: BeautifulSoup):
    """ Extract the title and reviews of a BeautifulSoup IMDB page """
    names = []
    reviews = []

    for elem in soup.find_all(class_='imdb-user-review'):
        name = elem.find(class_='title').get_text(strip=True)
        names.append(name)
        try:
            review = elem.find(class_="content").get_text(strip=True)
            reviews.append(review)
        except:
            continue

    return names, reviews


def scrape_reviews(urls: dict, driver_path: str = None, save: str = None) -> dict:
    """ Scrape all reviews from a single movie on IMDB and return a soup instance

    It needs to use Chrome driver as the "load more" button should be
    triggered multiple times in order to correctly load all reviews.

    Parameters:
    ----------
    driver_path : str, default None
        path to your chromedriver

    url : str
        The url to scrape

    Returns:
    --------
    soup : BeautifulSoup
        A BeautifulSoup instance of the entire page

    """

    all_reviews = {title: [] for title in urls}

    for movie_title in tqdm(urls, "Scraping IMDB Reviews"):

        # Instantiate driver
        driver = webdriver.Chrome(executable_path=driver_path)
        wait = WebDriverWait(driver, 10)

        # Prepare page
        driver.get(urls[movie_title])
        soup = BeautifulSoup(driver.page_source, 'lxml')
        time.sleep(3)

        # Press "load more" until the full page is loaded
        # Curtosey of: https://stackoverflow.com/questions/55527423/why-do-i-only-get-first-page-data-when-using-selenium
        while True:
            try:
                driver.find_element_by_css_selector("button#load-more-trigger").click()
                wait.until(EC.invisibility_of_element_located((By.CSS_SELECTOR, ".ipl-load-more__load-indicator")))
                soup = BeautifulSoup(driver.page_source, 'lxml')
            except Exception:
                break

        driver.quit()

        # Extract all reviews and their titles from the full page
        titles, reviews = extract_reviews(soup)
        all_reviews[movie_title] = [(review_title, movie_review) for review_title, movie_review in zip(titles, reviews)]

    if save:
        with open(f'../data/{save}_reviews.json', 'w') as f:
            json.dump(all_reviews, f)

    return all_reviews


def parse_arguments():
    parser = argparse.ArgumentParser(description='Scraper')
    parser.add_argument('--path', help='Chromedriver path', default="../drivers/chromedriver.exe")
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    # Extract titles from wiki
    disney = get_all_disney_titles()
    pixar = get_all_pixar_titles()

    # Extract IMDB movie urls
    disney_urls = scrape_imdb_urls(disney, save="disney")
    pixar_urls = scrape_imdb_urls(pixar, save="pixar")

    # Extract reviews
    pixar_reviews = scrape_reviews(pixar_urls, args.path, save="pixar")
    disney_reviews = scrape_reviews(disney_urls, args.path, save="disney")


if __name__ == "__main__":
    main()





