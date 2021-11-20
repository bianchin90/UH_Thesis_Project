# import
import spacy
import pandas as pd
import logging
from geopy import Nominatim

# set logger
logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(' UH MSc [Geoprocessing]')
logger.info(' modules imported correctly')

logging.getLogger('pandas').setLevel(logging.ERROR)

pd.options.mode.chained_assignment = None  # default='warn'

nlp = spacy.load("it_core_news_sm")

# running model on text and printing recognized entities
#raw = pd.read_excel('historical_data/historical_tweets_2016-09-01_2016-10-01.xlsx')

# istat df must be read in streamer
# istat = pd.read_excel('Georeferencing/Elenco-comuni-italiani.xls')
#
# istat = istat[['Denominazione (Italiana e straniera)',
#                "Denominazione dell'Unità territoriale sovracomunale (valida a fini statistici)",
#                'Denominazione Regione']]
# istat = istat.rename(columns={'Denominazione (Italiana e straniera)': 'city',
#                               "Denominazione dell'Unità territoriale sovracomunale (valida a fini statistici)": 'province',
#                               'Denominazione Regione': 'region'})
# istat['city'] = istat['city'].apply(lambda x: x.lower())

def find_city(cities_df, tweets):
    istat = cities_df
    # create empty df to host matching records
    geo_df = pd.DataFrame(columns=['city', 'lat', 'lon', 'tweets', 'magnitudo'])

    #abbreviations for maggnitude
    magnitude_abbr = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
                      'ML1', 'ML2', 'ML3', 'ML4', 'ML5', 'ML6', 'ML7', 'ML8', 'ML9']
    counter = 0
    for tweet in tweets:
        #clean tweets
        tweet = tweet.replace('\\n', ' ').replace('\\u2019', "'")

        #set default value for magnitude
        magnitudo = 'unknown'

        #look for magnitude
        for abbr in magnitude_abbr:
            if abbr in tweet:
                # replace magnitude abbreviation with 'magnitudo' + the following number
                tweet = tweet.replace(abbr, 'magnitudo {0}'.format(abbr[-1]))

        #extract the magnitude value, check on decimals is included
        mag_list = ['magnitudo', 'magnitudine']
        for elem in mag_list:
            if elem in tweet.lower() :
                try:
                    magnitudo = tweet.split(elem)[1].split(' ')[1]
                    if len(magnitudo) > 1:
                        if '.' in magnitudo:
                            magnitudo = magnitudo
                        else:
                            magnitudo = magnitudo[0] + '.' + magnitudo[1]
                except:
                    magnitudo = 'unknown'
        if not magnitudo.replace('.', '').isnumeric():
            magnitudo = 'unknown'

        #start real geoprocessing
        #perform entities recognition
        doc = nlp(tweet)

        places = []
        for ent in doc.ents:
            # select only LOC (localities)
            if ent.label_ == 'LOC':
                places.append(ent.text.lower())

        # match with italian cities database
        for location in places:
            location = location.replace('\\n', ' ').replace('\\u2019', "'")
            punc = '''+!()-[]{};:'"\,<>./?@#$%^&*_~'''
            for ele in location:
                if ele in punc:
                    location = location.replace(ele, "")
            match = istat[(istat["city"] == location)]

            #invoke geonominatim for coordinates
            for idx, city in match.iterrows():
                geolocator = Nominatim(user_agent="Earthquake_Detector")
                try:
                    location = geolocator.geocode(city['city'])
                except:
                    location = None
                if location is not None:
                    if 'italia' in location.address.lower():
                        #get only the first part of the official location name
                        #if the location is not already detected, set counter to 1
                        city = location.address.split(',')[0]
                        tweet_counter = 1

                        if (geo_df['city'] == city).any():
                            # if the location is in the list, increase counter by 1
                            mask = (geo_df['city'] == city)
                            geo_df['tweets'][mask] += 1
                            geo_df['magnitudo'][mask] = magnitudo

                        else:
                            geo_df.loc[len(geo_df)] = [city, location.latitude, location.longitude, tweet_counter, magnitudo]

                    counter += 1
    #return dataframe
    return geo_df
