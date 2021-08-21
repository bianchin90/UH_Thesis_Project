import json
import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from geopy.geocoders import Nominatim
from urllib.request import urlopen
import plotly.express as px
# from geopandas import GeoDataFrame
# from shapely.geometry import Point
import logging

# set logger
logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(' UH MSc [Geoprocessing]')
logger.info(' modules imported correctly')


# nltk.download('punkt')

# nlp = spacy.load("it_core_news_sm", disable=['parser', 'ner'])
# pd.set_option('display.width', 300)
# pd.set_option('display.max_columns', 20)

# nlp = spacy.load("it_core_news_sm", disable=['parser', 'ner'])
def find_city(cities_df, tweets):
    stop_words = stopwords.words('italian')
    # stop_words.extend(['nuova', 'santa', 'santo', 'nuovo', 'va'])

    # define geolocator for geocoding
    geolocator = Nominatim(user_agent="Earthquake_Detector")

    # istat df must be read in streamer
    # istat = pd.read_excel('Georeferencing/Elenco-comuni-italiani.xls')
    #
    # istat = istat[['Denominazione (Italiana e straniera)',
    #                "Denominazione dell'Unità territoriale sovracomunale (valida a fini statistici)",
    #                'Denominazione Regione']]
    # istat = istat.rename(columns={'Denominazione (Italiana e straniera)': 'city',
    #                               "Denominazione dell'Unità territoriale sovracomunale (valida a fini statistici)": 'province',
    #                               'Denominazione Regione': 'region'})
    #
    # istat['city'] = istat['city'].apply(lambda x: x.lower())
    istat = cities_df

    # test_tweets = ['nuova inaugurazione gelateria a roma', 'avvistato alieno a pinzolo',
    #                'il sindaco di opi promuove una nuova tassa sui rifiuti',
    #                'teramo: puppiniello arrestato dai carabinieri',
    #                'il piccoletto va in vacanza a trapani', 'tobia si è smarrito a roma']
    test_tweets = tweets

    # create empty df to host matching records
    geo_df = pd.DataFrame(columns=['city', 'lat', 'lon', 'tweets'])

    # iterate tweets
    for elem in test_tweets:
        # elem = sentence.lower()
        # process text (tokenization and stopwords)
        # initializing punctuations string
        punc = '''+!()-[]{};:'"\,<>./?@#$%^&*_~'''

        # Removing punctuations in string
        # Using loop + punctuation string
        for ele in elem:
            if ele in punc:
                elem = elem.replace(ele, "")
        elem = elem.replace('(', '').replace(')', '').replace(',', '')
        text_tokens = word_tokenize(elem)
        print(text_tokens)

        tokens_without_sw = [word for word in text_tokens if not word in stop_words]
        print(tokens_without_sw)

        # check if processed tweet contains any location information
        for term in tokens_without_sw:
            print(term)
            matches = istat['city'].str.contains(term).sum()
            if matches > 0:
                print(term)
                address = '{0}'.format(term)

                geolocator = Nominatim(user_agent="Earthquake_Detector")
                location = geolocator.geocode(address)
                print(location)
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

        # break
    print(geo_df)
    # geo_df.to_csv('Stream_Data/CitiesFound.csv')
    return  geo_df
    # geo_df['size'] = 2
    # geometry = [Point(xy) for xy in zip(geo_df.lon, geo_df.lat)]
    # geo_df = geo_df.drop(['lat', 'lon'], axis=1)
    # gdf = GeoDataFrame(geo_df, crs="EPSG:4326", geometry=geometry)
    # print(gdf)
    # exit()

    # https://plotly.github.io/plotly.py-docs/generated/plotly.express.scatter_mapbox.html


# riparti da qui
# https://www.youtube.com/watch?v=hSPmj7mK6ng
# https://github.com/Coding-with-Adam/Dash-by-Plotly/blob/master/Other/Dash_Introduction/intro.py
# https://plotly.com/python/choropleth-maps/
# https://plotly.com/python/map-configuration/
# https://medium.com/using-specialist-business-databases/creating-a-choropleth-map-using-geopandas-and-financial-data-c76419258746

# https://github.com/geopy/geopy
# https://medium.com/analytics-vidhya/how-to-generate-lat-and-long-coordinates-of-city-without-using-apis-25ebabcaf1d5


# integrate mapbox in dash
# https://www.youtube.com/watch?v=7R7VMSLwooo
# https://github.com/Coding-with-Adam/Dash-by-Plotly/blob/master/Dash_Interactive_Graphs/Scatter_mapbox/recycling.py

# update dash
# https://stackoverflow.com/questions/46075960/live-updating-only-the-data-in-dash-plotly
# https://stackoverflow.com/questions/54807868/how-to-fix-importerror-cannot-import-name-event-in-dash-from-plotly-python
# https://dash.plotly.com/live-updates

