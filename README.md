# Reviewer

<img src="https://github.com/MaartenGr/Reviewer/raw/master/images/results/result_0.png" height="300"/>

> Code for scraping IMDB reviews and applying NER and Sentiment Analyses 
>to extract Character popularity

**Reviewer** can be used to scrape user reviews from IMDB, generate word clouds based
on a custom class-based TF-IDF, and extract popular characters/actors from reviews
using a combination of Named Entity Recognition and Sentiment Analyses.   

## Table of Contents
<a name="toc"/></a>

1. [Instructions](#instructions)

    a. [Scrape](#instructions-scrape)
    
    b. [Word Cloud](#instructions-wordcloud)
    
    c. [c-TF-IDF](#instructions-ctfidf)

98. [Disney](#disney)

99. [Sources](#sources)

## 1. Instructions
[Back to ToC](#toc)
<a name="instructions"/></a>

I would advise you to start with the **Overview.ipynb** notebook for a good 
introduction before going to the command line. Moreover, scraping multiple movies
is actually preferred as it allows you to use the class-based TF-IDF. 

#### 1.a Scrape
<a name="instructions-scrape"/></a>

To scrape a single movie, simply run from the command line:
```commandline
python scraper.py --prefix lotr --url https://www.imdb.com/title/tt0120737/reviews?ref_=tt_ov_rt
```

Make sure to select the url of the review page of the movie you want to scrape. 
The `prefix` variable is the name used for saving the resulting .json file.  

#### 1.b Word Cloud
<a name="instructions-wordcloud"/></a>

Make sure that you save an image that you want to be used as a mask. 
It is important the background is white and the file saved as a .jpg

Then, after scraping the reviews, run the following from the command line:
```commandline
python scrape.py --path "data/lotr_count.json" --mask your_mask.jpg --pixels 1200
```

The data/lotr_count.json is the file saved after running the scraper. The name
thus depends on the prefix at the scrape stage. 

The result is something like this:

<img src="https://github.com/MaartenGr/Reviewer/raw/master/images/results/result_2.png" height="200"/>

#### 1.c c-TF-IDF
<a name="instructions-ctfidf"/></a>

The following formula can best be explained as a TF-IDF formula adopted for 
multiple classes by joining all documents per class. Thus, each class is converted 
to a single document instead of set of documents. Then, the frequency of words **t** 
are extracted for each class **i** and divided by the total number of words **w**. 

Next, the total, unjoined, number of documents across all classes **m** is divided by 
the total sum of word **i** across all classes. 

<img src="https://github.com/MaartenGr/Reviewer/raw/master/images/ctfidf.gif" height="100"/>


## 98. Disney
[Back to ToC](#toc)
<a name="disney"/></a>

Initially, this project was meant for me to be used only for Disney and Pixar movies 
(as I enjoy those very much), but eventually I generalized the code to be used for,
in principle, all movies. 

However, I also analyzed the most popular characters (by relative frequency in reviews)
and created a visualization of it below:

![image](https://github.com/MaartenGr/Reviewer/raw/master/images/disney_frequency.png)
      
## 99. Sources
[Back to ToC](#toc)
<a name="sources"/></a>

An overview of all sources used in this package (mainly images for masks). 

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

Opinion dataset:
- http://kavita-ganesan.com/opinosis/#.Wmljc6iWYow
- https://github.com/8horn/opinion-or-fact-sentence-classifier
