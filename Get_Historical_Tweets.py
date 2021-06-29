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
logger = logging.getLogger(' UH MSc [retrieve-historical-tweets]')
logger.info(' modules imported correctly')

desired_width=320

pd.set_option('display.width', desired_width)
#np.set_printoption(linewidth=desired_width)
pd.set_option('display.max_columns',10)

#define start scraping date Earthquake at L'Aquila
#start = add_months(datetime.datetime. strptime('2009-04-01', '%Y-%m-%d'), 0)
start = add_months(datetime.datetime. strptime('2012-01-01', '%Y-%m-%d'), 0)

#define temporal limit (next month)
today = datetime.date.today()
current_month =today.month
current_year = today.year

logger.info(' scraping started..')
while start != datetime.date(current_year, current_month + 1, 1):
    FROM  = start
    TO = add_months(start, 1)
    logger.info(' scraping from {0} to {1}'.format(FROM, TO))
    #search = '"terremoto" "magnitudo" since:2016-08-24  until:2016-08-26 lang:it -filter:replies -filter:retweets'
    search = '"terremoto" since:{0}  until:{1} lang:it -filter:replies -filter:retweets'.format(FROM, TO)
    #terremoto +lang:it +
    scraped_tweets = sntwitter.TwitterSearchScraper(search).get_items()

    # slicing the generator 1234
    sliced_scraped_tweets = itertools.islice(scraped_tweets, None)

    # convert to a DataFrame
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

    logger.info('saving to excel')
    df = df.applymap(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)
    #now = str(dt.datetime.now()).replace(' ', 'T').split('.')[0].replace(':', '')
    df.to_excel(new_path + "/historical_tweets_{0}_{1}.xlsx".format(FROM, TO), sheet_name='Sheet_name_1')
    start = TO



logger.info('..process completed')
