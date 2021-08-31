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
    geo_df = pd.DataFrame(columns=['city', 'lat', 'lon', 'tweets'])

    for tweet in tweets:
    # for index, row in raw.iterrows():
        # doc = nlp(row['content'])
        # print(row['content'])
        tweet = tweet.replace('\\n', ' ').replace('\\u2019', "'")
        doc = nlp(tweet)
        print(tweet)
        print(doc.ents)
        places = []
        for ent in doc.ents:
            print(ent.text, ent.start_char, ent.end_char, ent.label_)
            # select only LOC
            if ent.label_ == 'LOC':
                places.append(ent.text.lower())
        # match with istat
        for location in places:
            location = location.replace('\\n', ' ').replace('\\u2019', "'")
            punc = '''+!()-[]{};:'"\,<>./?@#$%^&*_~'''
            for ele in location:
                if ele in punc:
                    location = location.replace(ele, "")
            match = istat[(istat["city"] == location)]
            # if location != 'italia':
            #     match = istat[(istat["city"] == location) | (istat["city"].str.contains(location))]
            print(match)

            #invoke geonominatim for coordinates
            for idx, city in match.iterrows():
                geolocator = Nominatim(user_agent="Earthquake_Detector")
                location = geolocator.geocode(city['city'])
                if location is not None:
                    if 'italia' in location.address.lower():
                        city = location.address.split(',')[0]
                        tweet_counter = 1

                        if (geo_df['city'] == city).any():
                            # tweet_counter = geo_df.query('city=={0}'.format(city))['tweets'] +1
                            mask = (geo_df['city'] == city)
                            geo_df['tweets'][mask] += 1
                        else:
                            geo_df.loc[len(geo_df)] = [city, location.latitude, location.longitude, tweet_counter]

                    print('geocoded place: {0}, {1}, {2}'.format(location, location.latitude, location.longitude))
        #x = input(' Press any key to continue')
        #if x == 'stop':
        #    break
    #print(geo_df)
    return geo_df
# riparti da qui: https://techblog.smc.it/en/2020-12-11/nlp-ner
