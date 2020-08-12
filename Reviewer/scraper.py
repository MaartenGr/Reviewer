import os
import re
import json
import pandas as pd
from tqdm import tqdm

# Scraping
import requests
from urllib.parse import urljoin
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.selector import Selector
from bs4 import BeautifulSoup
from bs4.element import Tag


class Scraper:
    """
    Used to scrape IMDB reviews and parse them

    Parameters:
    -----------
    dir_path : str
        The path of the cwd, keep empty if there are no
        files to be saved in a parent dir

    prefix : str
        The prefix of the resulting saved files

    """
    def __init__(self, prefix: str, dir_path: str = ""):
        self.dir_path = dir_path
        self.prefix = prefix + "_"

    def scrape(self, urls: list) -> None:
        """ Scrape reviews from a list of urls (str) and saves them to data/ """
        process = CrawlerProcess(settings={
            "LOG_ENABLED": False,
            "FEEDS": {
                f"{self.dir_path}data/{self.prefix}reviews.json": {"format": "json"},
            },
        })
        process.crawl(IMDBSpider, urls=urls)
        process.start()

    def get_disney_urls(self) -> list:
        """ Scrape disney urls from wiki or load them from data/disney_urls.json """
        self.prefix = "disney_"
        if os.path.isfile(f"{self.dir_path}data/disney_urls.json"):
            with open(f"{self.dir_path}data/disney_urls.json", "r") as f:
                disney_urls = json.load(f)
                disney_urls = list(disney_urls.values())
        else:
            self.prefix = "disney_"
            disney = self.get_all_disney_titles()
            # disney = disney.iloc[:2]
            disney_urls = self.scrape_disney_imdb_urls(disney, save="disney")
            disney_urls = list(disney_urls.values())
        return disney_urls

    def parse_data(self):
        """ Parse saved reviews to save them in a nicer format """
        with open(f"{self.dir_path}data/{self.prefix}reviews.json", "r") as f:
            docs = json.load(f)

        titles = list(set([doc['title'] for doc in docs]))
        new_docs = {title: [] for title in titles}

        for doc in docs:
            parsed_doc = " ".join(doc["text"])
            new_docs[doc['title']].append(parsed_doc)

        # Save newly parsed data
        with open(f"{self.dir_path}data/{self.prefix}reviews.json", "w") as f:
            json.dump(new_docs, f)

    def get_all_disney_titles(self) -> pd.DataFrame:
        """ Get all Disney titles and their release dates """
        # Disney
        url = 'https://en.wikipedia.org/wiki/List_of_Walt_Disney_Animation_Studios_films'
        disney = pd.read_html(url, header=0)[1]
        disney['Year'] = disney.apply(lambda row: row['Release date'].split(",")[-1].strip(), 1)

        # Pixar
        url = "https://en.wikipedia.org/wiki/List_of_Pixar_films"
        pixar = pd.read_html(url, header=0)[0]
        pixar = pixar.loc[pixar.Film != "Released films", :]
        pixar = pixar.iloc[:22]
        pixar['Year'] = pixar.apply(lambda row: row['Release date'].split(",")[-1].strip(), 1)

        # Merge
        disney = disney.loc[:, ["Film", "Year"]]
        pixar = pixar.loc[:, ["Film", "Year"]]
        disney = disney.append(pixar)

        return disney

    def scrape_disney_imdb_urls(self, df: pd.DataFrame, save: str = None) -> dict:
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
                if self.match_years(result, search_term):
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
            with open(f'{self.dir_path}data/{save}_urls.json', 'w') as f:
                json.dump(urls, f)

        return urls

    def match_years(self, search_result: Tag, year: str) -> bool:
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


class IMDBSpider(scrapy.Spider):
    """
    Scrapy Spider for extracting reviews from IMDB
    Since new reviews can be loaded by clicking "load more",
    several urls have to be newly created.
    """
    name = "imdb"

    def __init__(self, urls, **kwargs):
        self.start_urls = urls
        super().__init__(**kwargs)

    def start_requests(self):
        """ Keeps track of the original url: Currently not used as title is extract from IMDB """
        for url in self.start_urls:
            yield scrapy.Request(url, meta={'orig_url': url})

    def parse(self, response):
        """ Extract reviews from IMDB and clicks on 'load more' to load more reviews """
        # ratings = response.xpath("//div[@class='ipl-ratings-bar']//span[@class='rating-other-user-rating']//"
        #                          "span[not(contains(@class, 'point-scale'))]/text()").getall()
        texts = response.xpath("//div[@class='text show-more__control']")

        try:
            title = response.xpath("//meta[@name='title']/@content")[0].extract().split("(")[0].strip()
        except:
            title = response.meta['title']

        for text in texts:
            text = text.extract()
            text = Selector(text=text)
            text = text.xpath("//div[@class='text show-more__control']/text()").extract()
            yield {
                "title": title,
                "text": text
            }

        key = response.css("div.load-more-data::attr(data-key)").get()
        orig_url = response.meta.get('orig_url', response.url)
        next_url = urljoin(orig_url, "reviews/_ajax?paginationKey={}".format(key))

        if key:
            yield scrapy.Request(next_url, meta={'orig_url': orig_url, "title": title})
