<p align="center">
<img src="https://github.com/MaartenGr/Reviewer/raw/master/images/results/result_0.png" height="400"/>
</p>

> Code for scraping IMDB reviews and applying NER and Sentiment Analyses 
>to extract Character popularity

**Reviewer** can be used to scrape user reviews from IMDB, generate word clouds based
on a custom class-based TF-IDF, and extract popular characters/actors from reviews
using a combination of Named Entity Recognition and Sentiment Analyses.   

<a name="toc"/></a>
## Table of Contents

1. [Instructions](#instructions)

    a. [Scrape](#instructions-scrape)
    
    b. [Word Cloud](#instructions-wordcloud)
    
    c. [Character Analysis](#instructions-character)
    
2. [c-TF-IDF](#ctf-idf)

3. [Disney](#disney)

4. [Sources](#sources)

<a name="instructions"/></a>
## 1. Instructions
[Back to ToC](#toc)

I would advise you to start with the **notebooks/Overview.ipynb** notebook for a good 
introduction before going to the command line. Moreover, scraping multiple movies
is actually preferred as it allows you to use the class-based TF-IDF.

Instead, you can **dowload** or **fork** this repo and start with the
instructions below. 

<a name="instructions-scrape"/></a>
#### 1.a Scrape

To scrape a single movie (e.g., Aladdin), simply run from the command line:
```commandline
python scraper.py --prefix aladdin --url https://www.imdb.com/title/tt0103639/reviews?ref_=tt_ov_rt
```

Make sure to select the url of the review page of the movie you want to scrape. 
The `prefix` variable is the name used for saving the resulting .json file.

Not only is the movie scraped, count data is also extracted if it is a single movie. If you want
to apply the class-based TF-IDF, I would suggest to follow the instructions at **notebooks/Overview.ipynb**.

<a name="instructions-wordcloud"/></a>
#### 1.b Word Cloud

Make sure that you save an image that you want to be used as a mask. 
It is important the background is white and the file saved as a .jpg. 

Then, after scraping the reviews, run the following from the command line:
```commandline
python scrape.py --path "data/aladdin_count.json" --mask your_mask.jpg --pixels 1200
```

The data/aladdin_count.json is the file saved after running the scraper. The name
thus depends on the prefix at the scrape stage. 

The result is something like this:

<p align="center">
<img src="https://github.com/MaartenGr/Reviewer/raw/master/images/wordclouds/result_1.png" height="200"/>
<img src="https://github.com/MaartenGr/Reviewer/raw/master/images/wordclouds/result_2.png" height="200"/>
<img src="https://github.com/MaartenGr/Reviewer/raw/master/images/wordclouds/result_3.png" height="200"/>
<img src="https://github.com/MaartenGr/Reviewer/raw/master/images/wordclouds/result_5.png" height="200"/>
</p>

<a name="instructions-characters"/></a>
#### 1.c Character Analysis
We want to extract, from the reviews, which characters and actors are often talked about. 
We start by using Named Entity Recognition to extract the entity "Person" from reviews. 
Then, in the sentence where the entity "Person" is found, we apply sentiment analysis to 
understand the sentiment about that character. In other words, we extract often talked about characters 
combined with how positive those characters are regarded.

To do this, I made use of **Named Entity Recognition** and **Sentiment Analysis** using pre-trained **BERT** models. 

After having scraped the review data, run the following from the command line:
```commandline
python char.py --movie Aladdin --extract True --fast True --prefix disney --rpath disney_reviews.json
```

The result will be the following visualization: 

<p align="center">
<img src="https://github.com/MaartenGr/Reviewer/raw/master/images/characters/aladdin_characters.png" height="400"/>
</p> 

<a name="ctf-idf"/></a>
## 2. Class-based TF-IDF
[Back to ToC](#toc)

This project uses a custom TF-IDF used for exploring words that are interesting based
on the differences between classes. In other words, words are only important if they are often
mentioned in one class, but not so much in all other classes. I call it a class-based TF-IDF (c-TF-IDF):

<p align="center">
<img src="https://github.com/MaartenGr/Reviewer/raw/master/images/ctfidf.gif" height="50"/>
</p>

The above formula can best be explained as a TF-IDF formula adopted for 
multiple classes by joining all documents per class. Thus, each class is converted 
to a single document instead of set of documents. Then, the frequency of words **t** 
are extracted for each class **i** and divided by the total number of words **w**. 

Next, the total, unjoined, number of documents across all classes **m** is divided by 
the total sum of word **i** across all classes.


<a name="disney"/></a>
## 3. Disney
[Back to ToC](#toc)

Initially, this project was meant for me to be used only for Disney and Pixar movies 
(as I enjoy those very much), but eventually I generalized the code to be used for,
in principle, all movies. 

However, I also analyzed the most popular characters (by relative frequency in reviews)
and created a visualization of it below:

<p align="center">
<img src="https://github.com/MaartenGr/Reviewer/raw/master/images/disney_frequency.png" height="400"/>
</p>

Moreover, you will find some Disney snippets here and there that I purposefully did not remove as
there were some manual fixes to get the visualizations working and the pipeline running. See
 **notebooks/Overview.ipynb** for more information on how to run that code. 

<a name="sources"/></a>      
## 4. Sources
[Back to ToC](#toc)

An overview of all sources used in this package (mainly images for masks).
All rights on these images belong to Disney, Pixar, and Marvel.  

<details>
<summary>Mask Images</summary>

* Aladdin - https://www.amazon.com/Aladdin-Official-Lifesize-Cardboard-Fan/dp/B07QSZ5GC9
* Coco - https://www.jing.fm/iclipt/mJTmmi/
* Avengers - https://besthqwallpapers.com/films/ironman-4k-superheroes-iron-man-white-background-38148
* Up - https://www.hiclipart.com/free-transparent-background-png-clipart-semci
* Toy Story 3 - https://pixar.fandom.com/wiki/Lots-o%27-Huggin%27_Bear
* Frozen - https://tvtropes.org/pmwiki/pmwiki.php/Characters/FrozenElsa
* Moana - https://brooklynactivemama.com/11/2016/obsessed-disneys-moana-free-moana-movie-printables.html
* Tangled - https://www.pngfuel.com/free-png/nfjxg
* Toy Story (Woody) - http://www.allocine.fr/evenements/pixar/chapitre2/
* Toy Story (Buzz) - https://heroes-and-villians.fandom.com/wiki/Buzz_Lightyear

</details>

<details>
<summary>Disney Popularity Images</summary>
 
* Simba - https://lionking.fandom.com/wiki/Simba
* Basil - https://disney.fandom.com/wiki/Basil_of_Baker_Street
* Mowgli - https://disney.fandom.com/wiki/Mowgli
* Woody - https://www.vhv.rs/viewpic/hbRoomw_woody-toy-story-png-png-download-toy-story/
* Woody - https://nl.disney.be/films/toy-story-4
* Carl - https://pixar.fandom.com/wiki/Carl_Fredricksen
* Mike - https://www.pngegg.com/en/png-emxle
* Belle - https://i.pinimg.com/550x/89/e6/29/89e629b622a929e9b2e1b825c34a3c71.jpg
* Rapunzel - https://picsart.com/i/318196865223211
* Ariel - https://princess.disney.com/ariel

</details>