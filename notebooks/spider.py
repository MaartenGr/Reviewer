import scrapy
from urllib.parse import urljoin


class RatingSpider(scrapy.Spider):
    name = "rate"
    start_urls = ["https://www.imdb.com/title/tt4695012/reviews?ref_=tt_ql_3"]

    def parse(self, response):
        texts = response.xpath("//div[@class='text show-more__control']/text()").getall()
        
        key = response.css("div.load-more-data::attr(data-key)").get()
        orig_url = response.meta.get('orig_url', response.url)
        next_url = urljoin(orig_url, "reviews/_ajax?paginationKey={}".format(key))
        if key:
            yield scrapy.Request(next_url, meta={'orig_url': orig_url})
        
        print(texts)
        break
#         break
#         with open("test.html", 'wb') as f:
#             f.write(response.body)