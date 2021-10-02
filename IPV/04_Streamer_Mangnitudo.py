import re
# Import the wordcloud library
import time

from gensim.models import CoherenceModel
from wordcloud import WordCloud
import gensim.corpora as corpora
import gensim
import nltk
import logging

nltk.download('stopwords')
import spacy
from nltk.corpus import stopwords
import datetime as dt
import recycle
from feel_it import EmotionClassifier, SentimentClassifier
import pandas as pd
import Geoprocessing_Test as Geoprocessing

# set logger
logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(' UH MSc [Streamer]')
logger.info(' modules imported correctly')

logging.getLogger('gensim').setLevel(logging.ERROR)
logging.getLogger('pandas').setLevel(logging.ERROR)

#############CODE FROM STREAMER
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None  # default='warn'


# def Sort(sub_li):
#     sub_li.sort(key=lambda x: x[1])
#     sub_li.reverse()
#     return sub_li

def most_frequent(List):
    return max(set(List), key = List.count)

# if __name__ == '__main__':
def process_data() :
    tot_emo = []
    # define dataframe containing istat cities
    istat = pd.read_excel("C:/Users/filip/PycharmProjects/UH_Thesis_Project/Georeferencing/Elenco-comuni-italiani.xls")

    istat = istat[['Denominazione (Italiana e straniera)',
                   "Denominazione dell'Unità territoriale sovracomunale (valida a fini statistici)",
                   'Denominazione Regione']]
    istat = istat.rename(columns={'Denominazione (Italiana e straniera)': 'city',
                                  "Denominazione dell'Unità territoriale sovracomunale (valida a fini statistici)": 'province',
                                  'Denominazione Regione': 'region'})
    istat['city'] = istat['city'].apply(lambda x: x.lower())

    #declare output dataframe for geoprocessing
    geo_df = pd.DataFrame(columns=['city', 'lat', 'lon', 'tweets', 'magnitudo'])

    # load model
    lda = gensim.models.LdaMulticore.load('C:/Users/filip/PycharmProjects/UH_Thesis_Project/final_model/final_model.model')

    # read sentiment dataset
    sentiment = pd.read_csv('C:/Users/filip/PycharmProjects/UH_Thesis_Project/SentimentData/Sentiment_words.csv', sep=';')

    # define emotion classifier
    emotion = EmotionClassifier()
    #input('Press any key to proceed')

    # Read data into papers
    logger.info(' Reading DF..')
    # raw = pd.read_excel('historical_data/historical_tweets_Test_Amatrice2.xlsx')
    raw = pd.read_excel('C:/Users/filip/PycharmProjects/UH_Thesis_Project/historical_data/historical_tweets_2016-09-01_2016-10-01.xlsx')
    for ix, ln in raw.iterrows():
        nowContent = recycle.remove_url(ln['content'])
        raw.at[ix, 'content'] = nowContent
    len_df = len(raw)

    # sort by date
    raw = raw.sort_values(by=['date'])

    # process in batches of 5 minutes
    # time_window = 30  # 12 hours: 720
    time_window = input(' Please express in minutes the desired time window: ')
    while not time_window.isnumeric():
        time_window = input(' Time window must be a numeric value: ')
    time_window = int(time_window)
    raw['date'] = pd.to_datetime(raw['date'])
    # print(papers[['date', 'content']])
    last = raw.date.max()
    start = raw.date.min()
    next = start + dt.timedelta(minutes=time_window)
    #print('{0}, {1}, {2}'.format(start, next, last))

    # define array x and y to plot
    x = []  # here you add the count of earthquakes detected
    y = []  # here you add the timeshift

    # declare dataframe for emotions perceived by population
    feelings = pd.DataFrame(columns=['feelings'])

    # only for test
    counter = 0
    while start < last:
        logger.info(' timeframe selected: from {0} to {1}'.format(start, next))
        df = raw[(raw['date'] >= start) & (raw['date'] < next)]

        # keep unnecessary columns
        papers = df[['content']]
        papers['content'] = papers['content'].replace(',', ' ').replace('  ', ' ')


        logger.info(' Processing textual attributes..')
        # Remove punctuation/lower casing
        papers['content_processed'] = \
            papers['content'].map(lambda x: re.sub('[,\\.!?]', '', str(x)))
        # Convert the titles to lowercase
        papers['content_processed'] = \
            papers['content_processed'].map(lambda x: x.lower())
        # Print out the first rows of papers
        papers['content_processed'].head()

        # Prepare data for LDA Analysis
        # start by tokenizing the text and removing stopwords.
        # Next, we convert the tokenized object into a corpus and dictionary

        logger.info(' Preparing data for LDA Analysis..')

        stop_words = stopwords.words('italian')
        stop_words.extend(['https', 'http', 'bqjkco', '\xe8', 'xe', 'xf', 'gi', 'pi', 'xec', 'tco', ','])

        data = papers.content_processed.values.tolist()
        data_words = list(recycle.sent_to_words(data))
        # remove stop words
        data_words = recycle.remove_stopwords(data_words)

        logger.info(' Building Bi- and Tri- grams..')
        # Build the bigram and trigram models
        # The two important arguments to Phrases are min_count and threshold.
        # The higher the values of these param, the harder it is for words to be combined
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        logger.info(' Removing stopwords..')
        # Remove Stop Words
        data_words_nostops = recycle.remove_stopwords(data_words)
        # Form Bigrams
        data_words_bigrams = recycle.make_bigrams(bigram_mod, data_words_nostops)
        # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
        nlp = spacy.load("it_core_news_sm", disable=['parser', 'ner'])
        # Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = recycle.lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        # logger.info(data_lemmatized[:1])

        # Create Dictionary
        id2word = corpora.Dictionary(data_lemmatized)
        # Create Corpus
        texts = data_lemmatized
        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        # feed
        # declare list containing predictions
        forecast = []
        logger.info(' Printing LDA predictions.. ')
        for body in corpus:
            vector = lda[body]
            new_v = sorted(vector[0], key=lambda x: x[1], reverse=True)
            if new_v[0][0] == 3:
                forecast.append('Earthquake')
            else:
                forecast.append('Other')

        # select only tweets marked as earthquake
        papers['forecast'] = forecast


        #################START TEST
        original = pd.read_csv('C:/Users/filip/PycharmProjects/UH_Thesis_Project/Georeferencing/earthquake_synonyms.csv', sep=',')

        synonyms = original.term.tolist()
        matching = pd.DataFrame(columns=['content'])

        for elem in synonyms:
            sel = papers[papers['content'].str.contains(elem, case=False, na=False)]
            sel = sel[['content']]
            matching = matching.append(sel)
        matching = matching.drop_duplicates()
        detected = papers.merge(matching, on='content')
        #print(detected.to_string())
        #################END TEST #questo pezzo qui non serve a nulla perchè annulla il lavoro dell'LDA. prova ad inserirlo nel geoprocesing cercando solo sisma/magnitudo/scossa

        if len(detected) > 0:
            #print(detected.to_string())
            #print(b)
            # run sentiment analysis
            logger.info(' Starting sentiment analysis..')
            sentiment_eval = recycle.perform_sentiment_analysis(sentiment_dataset=sentiment, tweet_dataset=detected)

            logger.info(' Computing emotions analysis..')
            emotion_content = sentiment_eval['content']
            emo = emotion.predict(emotion_content.to_list())
            sentiment_eval['emotions'] = emo

            for i, r in sentiment_eval.iterrows():
                tot_emo.append(r['sentiment_value'])
                severity = pd.DataFrame(columns=['severity'], data=tot_emo)
                #severity.at[0, 'severity'] = sum(tot_emo)/len(tot_emo)
                severity.to_csv('Stream_Data/Severity.csv', index=False)

                #scaled values
            #input('press any key to continue:')

            extra = {'feelings': emo}
            feelings = feelings.append(pd.DataFrame(extra))

            #run geoprocessing
            geoProc = Geoprocessing.find_city(cities_df=istat, tweets=detected['content'].tolist())
            if len(geoProc) > 0:
                #test
                for ix, location in geoProc.iterrows():
                    if (geo_df['city'] == location['city']).any():
                        # tweet_counter = geo_df.query('city=={0}'.format(city))['tweets'] +1
                        mask = (geo_df['city'] == location['city'])
                        geo_df['tweets'][mask] += location['tweets']
                        test_check = geo_df['magnitudo'][mask]
                        if test_check.iloc[0] == 'unknown':
                            geo_df['magnitudo'][mask] = location['magnitudo']
                    else:
                        geo_df.loc[len(geo_df)] = [location['city'], location['lat'], location['lon'], 1, location['magnitudo']]
                #end test
                # geo_df = geo_df.append(geoProc)
                geo_df.to_csv('Stream_Data/CitiesFound.csv', index=False)


        x.append(start)
        y.append(len(detected))

        #return statistics for this iteration
        logger.info('-----------------------------------------------------------------------------------------------')
        logger.info(' Overall statistics for range {0} - {1}'.format(start, next))
        logger.info(' Earthquake tweets found : {0}'.format(len(detected)))
        if len(geoProc) > 0:
#            logger.info('')
            logger.info(' Cities detected:')
            city_counter = 1
            geoProc['tweets'] = geoProc['tweets'].fillna(1)
            for ix, location in geoProc.iterrows():
                logger.info(' {0}) {1} ({2}, {3}). Magnitude: {4}. N° of tweets: {5}'.format(city_counter, location['city'], location['lat'], location['lon'], location['magnitudo'], location['tweets']))
                city_counter += 1
        else:
#            logger.info('')
            logger.info(' No cities detected in this range')
#        logger.info('')
        logger.info(' Most frequent emotion: {0}'.format(most_frequent(emo)))
        avg_sent = sentiment_eval['sentiment_value'].mean()
        if avg_sent < -0.8 :
            color_code ='Red'
        elif (avg_sent >= -0.8) and (avg_sent < -0.50) :
            color_code = 'Orange'
        elif avg_sent >= -0.5 :
            color_code = 'Yellow'
        logger.info(' Severity of tweets in this time window: {0}'.format(color_code))
        logger.info('-----------------------------------------------------------------------------------------------')


        # set next time window
        start = next
        next = next + dt.timedelta(minutes=time_window)
        counter += 1
        to_save = pd.DataFrame()
        to_save['X'] = x
        to_save['Y'] = y
        to_save.to_csv('Stream_Data/Earthquakes_Detection.csv', index=False)

        # pie chart (working)
        tt = pd.value_counts(feelings['feelings'])
        my_labels = feelings.feelings.unique()
        my_explode = (0, 0.1, 0, 0.1)
        feelings.to_csv('Stream_Data/SentimentResults.csv', index=False)

        # end test
        #print('counter: {0}'.format(counter))
        input('press any key to continue')
        #time.sleep(2)

    logger.info(' Streaming completed')

if __name__ == '__main__':
     process_data()
      #app.run_server(debug=False, dev_tools_hot_reload=True)

#riparti da qui https://community.plotly.com/t/solved-updating-server-side-app-data-on-a-schedule/6612
#devi capire come combinare il live streaming alla dashboard. prova ad eseguire i due processi in parallelo (magari con un ThreadPool).
# il data streamer salva i progressi a db (excel)
# le dash callback aggiornano i dati leggendo il db (excel aggiornato)


#sentiment analysis. calcolo della severity, riparti da qui https://www.earthdatascience.org/courses/use-data-open-source-python/intro-to-apis/analyze-tweet-sentiment-in-python/
# ci sta pure una funzione interessante per rimuovere gli url