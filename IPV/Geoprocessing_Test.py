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
    # geo_df = pd.DataFrame(columns=['city', 'lat', 'lon', 'tweets'])
    geo_df = pd.DataFrame(columns=['city', 'lat', 'lon', 'tweets', 'magnitudo'])

    magnitude_abbr = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
                      'ML1', 'ML2', 'ML3', 'ML4', 'ML5', 'ML6', 'ML7', 'ML8', 'ML9']
    counter = 0
    for tweet in tweets:
    # for index, row in raw.iterrows():
        # doc = nlp(row['content'])
        # print(row['content'])
        tweet = tweet.replace('\\n', ' ').replace('\\u2019', "'")
        magnitudo = 'unknown'
        #test with magnitudo
        for abbr in magnitude_abbr:
            if abbr in tweet:
                # print('found')
                # print(tweet)
                tweet = tweet.replace(abbr, 'magnitudo {0}'.format(abbr[-1]))
                # print(tweet)
                # input('press any key to continue: ')
        mag_list = ['magnitudo', 'magnitudine']
        for elem in mag_list:
            if elem in tweet.lower() :
                try:
                    magnitudo = tweet.split(elem)[1].split(' ')[1]
                    if len(magnitudo) > 1:
                        magnitudo = magnitudo[0] + '.' + magnitudo[1]
                except:
                    magnitudo = 'unknown'
        if not magnitudo.replace('.', '').isnumeric():
            magnitudo = 'unknown'
        # elif 'magnitudine' in tweet.lower() :
        #     magnitudo = tweet.split('magnitudine')[1].split(' ')[1]
        #     if len(magnitudo) > 1:
        #         magnitudo = magnitudo[0] + '.' + magnitudo[1]
        doc = nlp(tweet)
        #print(tweet)
        #print(doc.ents)
        places = []
        for ent in doc.ents:
            # print(ent.text, ent.start_char, ent.end_char, ent.label_)
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
            #print(match)

            #invoke geonominatim for coordinates
            for idx, city in match.iterrows():
                geolocator = Nominatim(user_agent="Earthquake_Detector")
                try:
                    location = geolocator.geocode(city['city'])
                except:
                    location = None
                if location is not None:
                    if 'italia' in location.address.lower():
                        city = location.address.split(',')[0]
                        tweet_counter = 1

                        if (geo_df['city'] == city).any():
                            # tweet_counter = geo_df.query('city=={0}'.format(city))['tweets'] +1
                            mask = (geo_df['city'] == city)
                            geo_df['tweets'][mask] += 1
                            geo_df['magnitudo'][mask] = magnitudo

                        else:
                            geo_df.loc[len(geo_df)] = [city, location.latitude, location.longitude, tweet_counter, magnitudo]

                    #logging.info('geocoded place in tweet n° {0}: {1} ({2}, {3}). Detected magnitude: {4}'.format(counter, city, location.latitude, location.longitude, magnitudo))
                    counter += 1
        # if magnitudo != 'unknown':
        #     print(geo_df)
        #     print(magnitudo)
        #     x = input(' Press any key to continue')
        #     if x == 'stop':
        #        break
    #print(geo_df)
    return geo_df
# riparti da qui: https://techblog.smc.it/en/2020-12-11/nlp-ner


# import recycle
# print('reading df')
# istat = pd.read_excel('Georeferencing/Elenco-comuni-italiani.xls')
# istat = istat[['Denominazione (Italiana e straniera)',
#                "Denominazione dell'Unità territoriale sovracomunale (valida a fini statistici)",
#                'Denominazione Regione']]
# istat = istat.rename(columns={'Denominazione (Italiana e straniera)': 'city',
#                               "Denominazione dell'Unità territoriale sovracomunale (valida a fini statistici)": 'province',
#                               'Denominazione Regione': 'region'})
# istat['city'] = istat['city'].apply(lambda x: x.lower())
#
# raw = pd.read_excel('historical_data/historical_tweets_Test_Amatrice2.xlsx')
# raw = raw.head(5000)
# # raw = pd.read_excel('historical_data/historical_tweets_2016-09-01_2016-10-01.xlsx')
# print('processing df')
# for ix, ln in raw.iterrows():
#     nowContent = recycle.remove_url(ln['content'])
#     raw.at[ix, 'content'] = nowContent
#
# gf = find_city(cities_df=istat, tweets=raw['content'].tolist())
#
# print(gf.to_string())
# gf.to_excel('Georeferencing/Georeferencing_Magnitudo.xlsx')

