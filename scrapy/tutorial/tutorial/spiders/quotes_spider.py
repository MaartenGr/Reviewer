# https://stackoverflow.com/questions/55808105/how-to-crawl-all-the-web-pages-of-a-website-i-could-crawl-only-2-web-pages

import json
import scrapy
from urllib.parse import urljoin


class RatingSpider(scrapy.Spider):
    name = "rate"
    start_urls = ["https://www.imdb.com/title/tt4695012/reviews"]

    def parse(self, response):
        ratings = response.xpath("//div[@class='ipl-ratings-bar']//span[@class='rating-other-user-rating']//span[not(contains(@class, 'point-scale'))]/text()").getall()
        texts = response.xpath("//div[@class='text show-more__control']/text()").getall()
        result_data = []
        for i in range(len(ratings)):
            yield {
                "ratings": int(ratings[i]),
                "review_text": texts[i]
            }
            
#         texts = response.xpath("//div[@class='text show-more__control']/text()").getall()
        key = response.css("div.load-more-data::attr(data-key)").get()
        orig_url = response.meta.get('orig_url', response.url)
        next_url = urljoin(orig_url, "reviews/_ajax?paginationKey={}".format(key))
        
        yield {"text": texts}
        
#         with open('result.json', "w") as data_file:    
#             old_data = json.load(data_file)
#         json.dumps({"text": texts})
        if key:
            yield scrapy.Request(next_url, meta={'orig_url': orig_url})
#         yield {"result": texts}
#         with open("test.json",'w') as f: 
#             json.dump(texts, f) 


