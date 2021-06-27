#note, sntwitter does not work with python 3.9, I used version 3.8.6
from snscrape.modules import twitter as sntwitter
import itertools
import pandas as pd
import datetime as dt
import os
import logging

#set logger
logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(' UH MSc [retrieve-historical-tweets]')
logger.info(' modules imported correctly')

desired_width=320

pd.set_option('display.width', desired_width)
#np.set_printoption(linewidth=desired_width)
pd.set_option('display.max_columns',10)

maxTweets = 100

#keyword = 'deprem'
#place = '5e02a0f0d91c76d2' #This geo_place string corresponds to İstanbul, Turkey on twitter.

#keyword = 'covid'
#place = '01fbe706f872cb32' This geo_place string corresponds to Washington DC on twitter.

#Open/create a file to append data to
#csvFile = open('place_result.csv', 'a', newline='', encoding='utf8')

#Use csv writer
#csvWriter = csv.writer(csvFile)
#csvWriter.writerow(['id','date','tweet',])

logger.info(' scraping started..')
#for i,tweet in enumerate(sntwitter.TwitterSearchScraper('deprem + place:5e02a0f0d91c76d2 + since:2020-10-31 until:2020-11-03 -filter:links -filter:replies').get_items()):
search = '"terremoto" "magnitudo" since:2016-08-24  until:2016-08-26 lang:it -filter:replies -filter:retweets'
#terremoto +lang:it +
scraped_tweets = sntwitter.TwitterSearchScraper(search).get_items()

# slicing the generator to keep only the first 100 tweets
sliced_scraped_tweets = itertools.islice(scraped_tweets, 10)

# convert to a DataFrame and keep only relevant columns
#df = pd.DataFrame(sliced_scraped_tweets)[['date', 'content']]
df = pd.DataFrame(sliced_scraped_tweets)

df['date'] = df['date'].astype(str)

directory = os.path.abspath(os.getcwd())
new_path = os.path.join(directory, "historical_data")

try:
    new_path = os.mkdir(new_path)
    logger.info(' Folder created')
except FileExistsError:
    logger.info(' Folder Already Exists')
    new_path = os.path.join(directory, "historical_data")

new_path = os.path.join(directory, "historical_data")


#print(df.dtypes)
now = str(dt.datetime.now()).replace(' ', 'T').split('.')[0].replace(':', '')
df.to_excel(new_path + "/historical_tweets_{0}.xlsx".format(now), sheet_name='Sheet_name_1')

logger.info('..process completed')
