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
import matplotlib.pyplot as plt
import recycle
from feel_it import EmotionClassifier, SentimentClassifier

import random as rd
from random import random

import pandas as pd
import numpy as np
import dash  # (version 1.0.0)
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import json
import plotly
import plotly.offline as py  # (version 4.4.1)
import plotly.graph_objs as go
import plotly.express as px
import dash_table
import dash_bootstrap_components as dbc

# set logger
logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(' UH MSc [Streamer]')
logger.info(' modules imported correctly')

#############CODE FROM STREAMER
pd.set_option('display.max_columns', None)


# def Sort(sub_li):
#     sub_li.sort(key=lambda x: x[1])
#     sub_li.reverse()
#     return sub_li


# if __name__ == '__main__':
def process_data() :
    # load model
    lda = gensim.models.LdaMulticore.load('final_model/final_model.model')

    # read sentiment dataset
    sentiment = pd.read_csv('SentimentData/Sentiment_words.csv', sep=';')

    # define emotion classifier
    emotion = EmotionClassifier()

    # Read data into papers
    logger.info(' Reading DF..')
    # raw = pd.read_excel('historical_data/historical_tweets_2012-05-01_2012-06-01.xlsx')
    raw = pd.read_excel('historical_data/historical_tweets_2016-09-01_2016-10-01.xlsx')
    len_df = len(raw)

    # sort by date
    raw = raw.sort_values(by=['date'])

    # process in batches of 5 minutes
    time_window = 60  # 12 hours: 720
    raw['date'] = pd.to_datetime(raw['date'])
    # print(papers[['date', 'content']])
    last = raw.date.max()
    start = raw.date.min()
    next = start + dt.timedelta(minutes=time_window)
    print('{0}, {1}, {2}'.format(start, next, last))

    # define array x and y to plot
    x = []  # here you add the count of earthquakes detected
    y = []  # here you add the timeshift

    # declare dataframe for emotions perceived by population
    feelings = pd.DataFrame(columns=['feelings'])

    # only for test
    counter = 0
    while start < last:
        print('timeframe selected: from {0} to {1}'.format(start, next))
        df = raw[(raw['date'] >= start) & (raw['date'] < next)]
        print(len(df))

        # keep unnecessary columns
        papers = df[['content']]
        # Print out the first rows of papers
        # papers.head()

        logger.info(' Processing textual attributes..')
        # Remove punctuation/lower casing
        papers['content_processed'] = \
            papers['content'].map(lambda x: re.sub('[,\\.!?]', '', str(x)))
        # Convert the titles to lowercase
        papers['content_processed'] = \
            papers['content_processed'].map(lambda x: x.lower())
        # Print out the first rows of papers
        papers['content_processed'].head()

        # Exploratory Analysis
        # To verify whether the preprocessing, we’ll make a word cloud using the wordcloud package to get a visual representation of most common words.
        # It is key to understanding the data and ensuring we are on the right track, and if any more preprocessing is necessary before training the model.

        logger.info(' Generating WordCloud..')
        # Join the different processed titles together.
        long_string = ','.join(list(papers['content_processed'].values))
        # Create a WordCloud object
        wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue',
                              width=800, height=400)
        # Generate a word cloud
        # wordcloud.generate(long_string)
        # Visualize the word cloud
        # wordcloud.to_image()

        # Prepare data for LDA Analysis
        # start by tokenizing the text and removing stopwords.
        # Next, we convert the tokenized object into a corpus and dictionary

        logger.info(' Preparing data for LDA Analysis..')

        stop_words = stopwords.words('italian')
        stop_words.extend(['https', 'http', 'bqjkco', '\xe8', 'xe', 'xf', 'gi', 'pi', 'xec', 'tco'])

        data = papers.content_processed.values.tolist()
        data_words = list(recycle.sent_to_words(data))
        # remove stop words
        data_words = recycle.remove_stopwords(data_words)
        # logger.info(data_words[:1][0][:30])

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
        logger.info('printing LDA predictions.. ')
        for body in corpus:
            vector = lda[body]
            # new_v = Sort(vector[0])
            new_v = sorted(vector[0], key=lambda x: x[1], reverse=True)
            if new_v[0][0] == 3:
                print(' Earthquake detected')
                forecast.append('Earthquake')
            else:
                forecast.append('Other')

        # select only tweets marked as earthquake
        papers['forecast'] = forecast
        detected = papers[papers['forecast'] == 'Earthquake']

        logger.info(' Earthquake tweets detected: {0}'.format(len(detected)))

        if len(detected) > 0:
            # run sentiment analysis
            sentiment_eval = recycle.perform_sentiment_analysis(sentiment_dataset=sentiment, tweet_dataset=detected)
            print(' Sentiment analysis results')
            print(sentiment_eval['sentiment_value'].min())
            print(sentiment_eval['sentiment_value'].max())

            print(sentiment_eval['sentiment_unprocessed'].min())
            print(sentiment_eval['sentiment_unprocessed'].max())

            print(' computing emotions analysis..')
            emotion_content = sentiment_eval['content']
            emo = emotion.predict(emotion_content.to_list())
            # my_emo = set(emo)
            sentiment_eval['emotions'] = emo
            print(emo)

            extra = {'feelings': emo}
            feelings = feelings.append(pd.DataFrame(extra))

        x.append(start)
        # y.append(len(n_detection))
        y.append(forecast.count('Earthquake'))

        # set next time window
        start = next
        next = next + dt.timedelta(minutes=time_window)
        counter += 1
        # keep only last n records for line plot (fresherst data)
        N = 15
        # using list slicing
        # Get last N elements from list
        # if len(x) > N:
        #     x = x[-N:]
        #     y = y[-N:]
        #     plt.clf()
        to_save = pd.DataFrame()
        to_save['X'] = x
        to_save['Y'] = y
        to_save.to_csv('Stream_Data/Earthquakes_Detection.csv', index=False)

        # pie chart (working)
        tt = pd.value_counts(feelings['feelings'])
        my_labels = feelings.feelings.unique()
        my_explode = (0, 0.1, 0, 0.1)

        # end test
        print('counter: {0}'.format(counter))
        time.sleep(2)

        if counter == 150:
            print(tt)
            # print(my_labels)
            break
    print('done')

if __name__ == '__main__':
     process_data()
      #app.run_server(debug=False, dev_tools_hot_reload=True)

#riparti da qui https://community.plotly.com/t/solved-updating-server-side-app-data-on-a-schedule/6612
#devi capire come combinare il live streaming alla dashboard. prova ad eseguire i due processi in parallelo (magari con un ThreadPool).
# il data streamer salva i progressi a db (excel)
# le dash callback aggiornano i dati leggendo il db (excel aggiornato)