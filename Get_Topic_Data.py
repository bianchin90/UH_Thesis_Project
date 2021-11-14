#note, sntwitter does not work with python 3.9, I used version 3.8.6
from snscrape.modules import twitter as sntwitter
import itertools
import pandas as pd
import os
import logging
import datetime
import calendar

def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year,month)[1])
    return datetime.date(year, month, day)


#set logger
logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(' UH MSc [retrieve-other-Twitter-Topics]')
logger.info(' modules imported correctly')

desired_width=320

pd.set_option('display.width', desired_width)
#np.set_printoption(linewidth=desired_width)
pd.set_option('display.max_columns',10)


#define a list of topics
#topics = ['#EURO2020', '#G20', '#vaccini', '#ddlzan', '#ferragni', '#sanremo', '#maneskin']
topics = ['#clima', '#sicilia', '#enricoletta']


for topic in topics:
    logger.info(' scraping topic {0}'.format(topic))
    #search = '"terremoto" "magnitudo" since:2016-08-24  until:2016-08-26 lang:it -filter:replies -filter:retweets'
    # search = '"{0}" since:2021-02-22  until:2021-07-01 lang:it -filter:replies -filter:retweets'.format(topic)
    search = '"{0}" since:2019-01-30  until:2019-02-01 lang:it -filter:replies -filter:retweets'.format(topic)
    #terremoto +lang:it +
    # max_num = 200000
    max_num = 200
    scraped_tweets = sntwitter.TwitterSearchScraper(search).get_items()

    # slicing the generator 1234
    sliced_scraped_tweets = itertools.islice(scraped_tweets, max_num)

    # convert to a DataFrame
    df = pd.DataFrame(sliced_scraped_tweets)

    df['date'] = df['date'].astype(str)
    print(len(df))
    #print(b)

    directory = os.path.abspath(os.getcwd())
    new_path = os.path.join(directory, "training_topics")

    try:
        new_path = os.mkdir(new_path)
        logger.info(' Folder created')
    except FileExistsError:
        logger.info(' Folder Already Exists')
        new_path = os.path.join(directory, "training_topics")

    new_path = os.path.join(directory, "training_topics")

    logger.info('saving to excel')
    df = df.applymap(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)
    #now = str(dt.datetime.now()).replace(' ', 'T').split('.')[0].replace(':', '')
    df.to_excel(new_path + "/tweets_{0}.xlsx".format(topic), sheet_name='Sheet_name_1')



logger.info('..process completed')
