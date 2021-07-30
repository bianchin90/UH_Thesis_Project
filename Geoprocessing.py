import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#import overpass
from geopy import geocoders

nltk.download('punkt')

nlp = spacy.load("it_core_news_sm", disable=['parser', 'ner'])
pd.set_option('display.width', 300)
pd.set_option('display.max_columns',20)


nlp = spacy.load("it_core_news_sm", disable=['parser', 'ner'])
stop_words = stopwords.words('italian')
stop_words.extend(['nuova', 'santa', 'santo', 'nuovo', 'va'])
#api = overpass.API()
gn = geocoders.GeoNames()

istat = pd.read_excel('Georeferencing/Elenco-comuni-italiani.xls')

istat = istat[['Denominazione (Italiana e straniera)', "Denominazione dell'Unità territoriale sovracomunale (valida a fini statistici)", 'Denominazione Regione' ]]
istat = istat.rename(columns= {'Denominazione (Italiana e straniera)' : 'city', "Denominazione dell'Unità territoriale sovracomunale (valida a fini statistici)": 'province',
                               'Denominazione Regione' : 'region'})


istat['city'] = istat['city'].apply(lambda x: x.lower())


test_tweets = ['nuova inaugurazione gelateria a roma', 'avvistato orso a pinzolo', 'il sindaco di opi promuove una nuova tassa sui rifiuti', 'teramo: puppiniello arrestato dai carabinieri',
               'il piccoletto va in vacanza a santa marinella']


#create empty df to host matching records
geo_df = pd.DataFrame()

#iterate tweets
for elem in test_tweets:

    #process text (tokenization and stopwords)
    text_tokens = word_tokenize(elem)
    print(text_tokens)

    tokens_without_sw = [word for word in text_tokens if not word in stop_words]
    print(tokens_without_sw)

    #check if processed tweet contains any location information
    for term in tokens_without_sw :
        matches = istat['city'].str.contains(term).sum()
        if matches > 0 :
            print(term)
            #response = api.get('node["city"="{0}"]'.format(term))
            response = gn.geocode('{0}, Italy'.format(term))
            print(response)
    break


    #riparti da qui https://github.com/geopy/geopy
    #hai trovato la città, ora devi beccare le coordinate